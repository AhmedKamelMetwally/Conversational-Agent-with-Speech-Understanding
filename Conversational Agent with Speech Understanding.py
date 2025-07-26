from langchain_core.tools import StructuredTool

from langflow.base.agents.agent import LCToolsAgentComponent
from langflow.base.agents.events import ExceptionWithMessageError
from langflow.base.models.model_input_constants import (
    ALL_PROVIDER_FIELDS,
    MODEL_DYNAMIC_UPDATE_FIELDS,
    MODEL_PROVIDERS,
    MODEL_PROVIDERS_DICT,
    MODELS_METADATA,
)
from langflow.base.models.model_utils import get_model_name
from langflow.components.helpers.current_date import CurrentDateComponent
from langflow.components.helpers.memory import MemoryComponent
from langflow.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from langflow.custom.custom_component.component import _get_component_toolkit
from langflow.custom.utils import update_component_build_config
from langflow.field_typing import Tool
from langflow.io import BoolInput, DropdownInput, IntInput, MultilineInput, Output
from langflow.logging import logger
from langflow.schema.dotdict import dotdict
from langflow.schema.message import Message

import assemblyai as aai
from loguru import logger

from langflow.custom.custom_component.component import Component
from langflow.field_typing.range_spec import RangeSpec
from langflow.io import DataInput, FloatInput, Output, SecretStrInput
from langflow.schema.data import Data
from pathlib import Path



def set_advanced_true(component_input):
    component_input.advanced = True
    return component_input


MODEL_PROVIDERS_LIST = ["Anthropic", "Google Generative AI", "Groq", "OpenAI"]


class AgentComponent(ToolCallingAgentComponent):
    display_name: str = "Agent"
    description: str = "Define the agent's instructions, then enter a task to complete using tools."
    documentation: str = "https://docs.langflow.org/agents"
    icon = "bot"
    beta = False
    name = "Agent"

    memory_inputs = [set_advanced_true(component_input) for component_input in MemoryComponent().inputs]

    inputs = [
        DropdownInput(
            name="agent_llm",
            display_name="Model Provider",
            info="The provider of the language model that the agent will use to generate responses.",
            options=[*MODEL_PROVIDERS_LIST, "Custom"],
            value="OpenAI",
            real_time_refresh=True,
            input_types=[],
            options_metadata=[MODELS_METADATA[key] for key in MODEL_PROVIDERS_LIST] + [{"icon": "brain"}],
        ),
        *MODEL_PROVIDERS_DICT["OpenAI"]["inputs"],
        MultilineInput(
            name="system_prompt",
            display_name="Agent Instructions",
            info="System Prompt: Initial instructions and context provided to guide the agent's behavior.",
            value="You are a helpful assistant that can use tools to answer questions and perform tasks.",
            advanced=False,
        ),
        IntInput(
            name="n_messages",
            display_name="Number of Chat History Messages",
            value=100,
            info="Number of chat history messages to retrieve.",
            advanced=True,
            show=True,
        ),
        *LCToolsAgentComponent._base_inputs,
        # removed memory inputs from agent component
        # *memory_inputs,
        BoolInput(
            name="add_current_date_tool",
            display_name="Current Date",
            advanced=True,
            info="If true, will add a tool to the agent that returns the current date.",
            value=True,
        ),
    ]
    outputs = [Output(name="response", display_name="Response", method="message_response")]

    async def message_response(self) -> Message:
        try:
            # Get LLM model and validate
            llm_model, display_name = self.get_llm()
            if llm_model is None:
                msg = "No language model selected. Please choose a model to proceed."
                raise ValueError(msg)
            self.model_name = get_model_name(llm_model, display_name=display_name)

            # Get memory data
            self.chat_history = await self.get_memory_data()
            if isinstance(self.chat_history, Message):
                self.chat_history = [self.chat_history]

            # Add current date tool if enabled
            if self.add_current_date_tool:
                if not isinstance(self.tools, list):  # type: ignore[has-type]
                    self.tools = []
                current_date_tool = (await CurrentDateComponent(**self.get_base_args()).to_toolkit()).pop(0)
                if not isinstance(current_date_tool, StructuredTool):
                    msg = "CurrentDateComponent must be converted to a StructuredTool"
                    raise TypeError(msg)
                self.tools.append(current_date_tool)
            # note the tools are not required to run the agent, hence the validation removed.

            # Set up and run agent
            self.set(
                llm=llm_model,
                tools=self.tools or [],
                chat_history=self.chat_history,
                input_value=self.input_value,
                system_prompt=self.system_prompt,
            )
            agent = self.create_agent_runnable()
            return await self.run_agent(agent)

        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"{type(e).__name__}: {e!s}")
            raise
        except ExceptionWithMessageError as e:
            logger.error(f"ExceptionWithMessageError occurred: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e!s}")
            raise

    async def get_memory_data(self):
        # TODO: This is a temporary fix to avoid message duplication. We should develop a function for this.
        messages = (
            await MemoryComponent(**self.get_base_args())
            .set(session_id=self.graph.session_id, order="Ascending", n_messages=self.n_messages)
            .retrieve_messages()
        )
        return [
            message for message in messages if getattr(message, "id", None) != getattr(self.input_value, "id", None)
        ]

    def get_llm(self):
        if not isinstance(self.agent_llm, str):
            return self.agent_llm, None

        try:
            provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
            if not provider_info:
                msg = f"Invalid model provider: {self.agent_llm}"
                raise ValueError(msg)

            component_class = provider_info.get("component_class")
            display_name = component_class.display_name
            inputs = provider_info.get("inputs")
            prefix = provider_info.get("prefix", "")

            return self._build_llm_model(component_class, inputs, prefix), display_name

        except Exception as e:
            logger.error(f"Error building {self.agent_llm} language model: {e!s}")
            msg = f"Failed to initialize language model: {e!s}"
            raise ValueError(msg) from e

    def _build_llm_model(self, component, inputs, prefix=""):
        model_kwargs = {}
        for input_ in inputs:
            if hasattr(self, f"{prefix}{input_.name}"):
                model_kwargs[input_.name] = getattr(self, f"{prefix}{input_.name}")
        return component.set(**model_kwargs).build_model()

    def set_component_params(self, component):
        provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
        if provider_info:
            inputs = provider_info.get("inputs")
            prefix = provider_info.get("prefix")
            model_kwargs = {input_.name: getattr(self, f"{prefix}{input_.name}") for input_ in inputs}

            return component.set(**model_kwargs)
        return component

    def delete_fields(self, build_config: dotdict, fields: dict | list[str]) -> None:
        """Delete specified fields from build_config."""
        for field in fields:
            build_config.pop(field, None)

    def update_input_types(self, build_config: dotdict) -> dotdict:
        """Update input types for all fields in build_config."""
        for key, value in build_config.items():
            if isinstance(value, dict):
                if value.get("input_types") is None:
                    build_config[key]["input_types"] = []
            elif hasattr(value, "input_types") and value.input_types is None:
                value.input_types = []
        return build_config

    async def update_build_config(
        self, build_config: dotdict, field_value: str, field_name: str | None = None
    ) -> dotdict:
        # Iterate over all providers in the MODEL_PROVIDERS_DICT
        # Existing logic for updating build_config
        if field_name in ("agent_llm",):
            build_config["agent_llm"]["value"] = field_value
            provider_info = MODEL_PROVIDERS_DICT.get(field_value)
            if provider_info:
                component_class = provider_info.get("component_class")
                if component_class and hasattr(component_class, "update_build_config"):
                    # Call the component class's update_build_config method
                    build_config = await update_component_build_config(
                        component_class, build_config, field_value, "model_name"
                    )

            provider_configs: dict[str, tuple[dict, list[dict]]] = {
                provider: (
                    MODEL_PROVIDERS_DICT[provider]["fields"],
                    [
                        MODEL_PROVIDERS_DICT[other_provider]["fields"]
                        for other_provider in MODEL_PROVIDERS_DICT
                        if other_provider != provider
                    ],
                )
                for provider in MODEL_PROVIDERS_DICT
            }
            if field_value in provider_configs:
                fields_to_add, fields_to_delete = provider_configs[field_value]

                # Delete fields from other providers
                for fields in fields_to_delete:
                    self.delete_fields(build_config, fields)

                # Add provider-specific fields
                if field_value == "OpenAI" and not any(field in build_config for field in fields_to_add):
                    build_config.update(fields_to_add)
                else:
                    build_config.update(fields_to_add)
                # Reset input types for agent_llm
                build_config["agent_llm"]["input_types"] = []
            elif field_value == "Custom":
                # Delete all provider fields
                self.delete_fields(build_config, ALL_PROVIDER_FIELDS)
                # Update with custom component
                custom_component = DropdownInput(
                    name="agent_llm",
                    display_name="Language Model",
                    options=[*sorted(MODEL_PROVIDERS), "Custom"],
                    value="Custom",
                    real_time_refresh=True,
                    input_types=["LanguageModel"],
                    options_metadata=[MODELS_METADATA[key] for key in sorted(MODELS_METADATA.keys())]
                    + [{"icon": "brain"}],
                )
                build_config.update({"agent_llm": custom_component.to_dict()})
            # Update input types for all fields
            build_config = self.update_input_types(build_config)

            # Validate required keys
            default_keys = [
                "code",
                "_type",
                "agent_llm",
                "tools",
                "input_value",
                "add_current_date_tool",
                "system_prompt",
                "agent_description",
                "max_iterations",
                "handle_parsing_errors",
                "verbose",
            ]
            missing_keys = [key for key in default_keys if key not in build_config]
            if missing_keys:
                msg = f"Missing required keys in build_config: {missing_keys}"
                raise ValueError(msg)
        if (
            isinstance(self.agent_llm, str)
            and self.agent_llm in MODEL_PROVIDERS_DICT
            and field_name in MODEL_DYNAMIC_UPDATE_FIELDS
        ):
            provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
            if provider_info:
                component_class = provider_info.get("component_class")
                component_class = self.set_component_params(component_class)
                prefix = provider_info.get("prefix")
                if component_class and hasattr(component_class, "update_build_config"):
                    # Call each component class's update_build_config method
                    # remove the prefix from the field_name
                    if isinstance(field_name, str) and isinstance(prefix, str):
                        field_name = field_name.replace(prefix, "")
                    build_config = await update_component_build_config(
                        component_class, build_config, field_value, "model_name"
                    )
        return dotdict({k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in build_config.items()})

    async def _get_tools(self) -> list[Tool]:
        component_toolkit = _get_component_toolkit()
        tools_names = self._build_tools_names()
        agent_description = self.get_tool_description()
        # TODO: Agent Description Depreciated Feature to be removed
        description = f"{agent_description}{tools_names}"
        tools = component_toolkit(component=self).get_tools(
            tool_name="Call_Agent", tool_description=description, callbacks=self.get_langchain_callbacks()
        )
        if hasattr(self, "tools_metadata"):
            tools = component_toolkit(component=self, metadata=self.tools_metadata).update_tools_metadata(tools=tools)
        return tools
class AssemblyAITranscriptionJobCreator(Component):
    display_name = "AssemblyAI Start Transcript"
    description = "Create a transcription job for an audio file using AssemblyAI with advanced options"
    documentation = "https://www.assemblyai.com/docs"
    icon = "AssemblyAI"

    inputs = [
        SecretStrInput(
            name="api_key",
            display_name="Assembly API Key",
            info="Your AssemblyAI API key. You can get one from https://www.assemblyai.com/",
            required=True,
        ),
        FileInput(
            name="audio_file",
            display_name="Audio File",
            file_types=[
                "3ga",
                "8svx",
                "aac",
                "ac3",
                "aif",
                "aiff",
                "alac",
                "amr",
                "ape",
                "au",
                "dss",
                "flac",
                "flv",
                "m4a",
                "m4b",
                "m4p",
                "m4r",
                "mp3",
                "mpga",
                "ogg",
                "oga",
                "mogg",
                "opus",
                "qcp",
                "tta",
                "voc",
                "wav",
                "wma",
                "wv",
                "webm",
                "mts",
                "m2ts",
                "ts",
                "mov",
                "mp2",
                "mp4",
                "m4p",
                "m4v",
                "mxf",
            ],
            info="The audio file to transcribe",
            required=True,
        ),
        MessageTextInput(
            name="audio_file_url",
            display_name="Audio File URL",
            info="The URL of the audio file to transcribe (Can be used instead of a File)",
            advanced=True,
        ),
        DropdownInput(
            name="speech_model",
            display_name="Speech Model",
            options=[
                "best",
                "nano",
            ],
            value="best",
            info="The speech model to use for the transcription",
            advanced=True,
        ),
        BoolInput(
            name="language_detection",
            display_name="Automatic Language Detection",
            info="Enable automatic language detection",
            advanced=True,
        ),
        MessageTextInput(
            name="language_code",
            display_name="Language",
            info=(
                """
            The language of the audio file. Can be set manually if automatic language detection is disabled.
            See https://www.assemblyai.com/docs/getting-started/supported-languages """
                "for a list of supported language codes."
            ),
            advanced=True,
        ),
        BoolInput(
            name="speaker_labels",
            display_name="Enable Speaker Labels",
            info="Enable speaker diarization",
        ),
        MessageTextInput(
            name="speakers_expected",
            display_name="Expected Number of Speakers",
            info="Set the expected number of speakers (optional, enter a number)",
            advanced=True,
        ),
        BoolInput(
            name="punctuate",
            display_name="Punctuate",
            info="Enable automatic punctuation",
            advanced=True,
            value=True,
        ),
        BoolInput(
            name="format_text",
            display_name="Format Text",
            info="Enable text formatting",
            advanced=True,
            value=True,
        ),
    ]

    outputs = [
        Output(display_name="Transcript ID", name="transcript_id", method="create_transcription_job"),
    ]

    def create_transcription_job(self) -> Data:
        aai.settings.api_key = self.api_key

        # Convert speakers_expected to int if it's not empty
        speakers_expected = None
        if self.speakers_expected and self.speakers_expected.strip():
            try:
                speakers_expected = int(self.speakers_expected)
            except ValueError:
                self.status = "Error: Expected Number of Speakers must be a valid integer"
                return Data(data={"error": "Error: Expected Number of Speakers must be a valid integer"})

        language_code = self.language_code or None

        config = aai.TranscriptionConfig(
            speech_model=self.speech_model,
            language_detection=self.language_detection,
            language_code=language_code,
            speaker_labels=self.speaker_labels,
            speakers_expected=speakers_expected,
            punctuate=self.punctuate,
            format_text=self.format_text,
        )

        audio = None
        if self.audio_file:
            if self.audio_file_url:
                logger.warning("Both an audio file an audio URL were specified. The audio URL was ignored.")

            # Check if the file exists
            if not Path(self.audio_file).exists():
                self.status = "Error: Audio file not found"
                return Data(data={"error": "Error: Audio file not found"})
            audio = self.audio_file
        elif self.audio_file_url:
            audio = self.audio_file_url
        else:
            self.status = "Error: Either an audio file or an audio URL must be specified"
            return Data(data={"error": "Error: Either an audio file or an audio URL must be specified"})

        try:
            transcript = aai.Transcriber().submit(audio, config=config)
        except Exception as e:  # noqa: BLE001
            logger.opt(exception=True).debug("Error submitting transcription job")
            self.status = f"An error occurred: {e}"
            return Data(data={"error": f"An error occurred: {e}"})

        if transcript.error:
            self.status = transcript.error
            return Data(data={"error": transcript.error})
        result = Data(data={"transcript_id": transcript.id})
        self.status = result
        return result
class AssemblyAITranscriptionJobPoller(Component):
    display_name = "AssemblyAI Poll Transcript"
    description = "Poll for the status of a transcription job using AssemblyAI"
    documentation = "https://www.assemblyai.com/docs"
    icon = "AssemblyAI"

    inputs = [
        SecretStrInput(
            name="api_key",
            display_name="Assembly API Key",
            info="Your AssemblyAI API key. You can get one from https://www.assemblyai.com/",
            required=True,
        ),
        DataInput(
            name="transcript_id",
            display_name="Transcript ID",
            info="The ID of the transcription job to poll",
            required=True,
        ),
        FloatInput(
            name="polling_interval",
            display_name="Polling Interval",
            value=3.0,
            info="The polling interval in seconds",
            advanced=True,
            range_spec=RangeSpec(min=3, max=30),
        ),
    ]

    outputs = [
        Output(display_name="Transcription Result", name="transcription_result", method="poll_transcription_job"),
    ]

    def poll_transcription_job(self) -> Data:
        """Polls the transcription status until completion and returns the Data."""
        aai.settings.api_key = self.api_key
        aai.settings.polling_interval = self.polling_interval

        # check if it's an error message from the previous step
        if self.transcript_id.data.get("error"):
            self.status = self.transcript_id.data["error"]
            return self.transcript_id

        try:
            transcript = aai.Transcript.get_by_id(self.transcript_id.data["transcript_id"])
        except Exception as e:  # noqa: BLE001
            error = f"Getting transcription failed: {e}"
            logger.opt(exception=True).debug(error)
            self.status = error
            return Data(data={"error": error})

        if transcript.status == aai.TranscriptStatus.completed:
            json_response = transcript.json_response
            text = json_response.pop("text", None)
            utterances = json_response.pop("utterances", None)
            transcript_id = json_response.pop("id", None)
            sorted_data = {"text": text, "utterances": utterances, "id": transcript_id}
            sorted_data.update(json_response)
            data = Data(data=sorted_data)
            self.status = data
            return data
        self.status = transcript.error
        return Data(data={"error": transcript.error})
