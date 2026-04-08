"""User simulator agent.

Simulates user behavior based on a persona and scenario,
generating realistic user messages in conversations.
"""

from __future__ import annotations

from typing import Any

from evaluatorq.simulation.agents.base import AgentConfig, BaseAgent
from evaluatorq.simulation.types import ChatMessage

DEFAULT_USER_SIMULATOR_PROMPT = """You are a user simulator. Your role is to simulate realistic user behavior in a conversation with an AI agent.

You will be given:
1. A persona describing who you are and how you behave
2. A scenario describing your goal and context

Your task:
- Generate realistic user messages based on your persona and scenario
- Stay in character throughout the conversation
- Work towards achieving your goal naturally
- React authentically to the agent's responses
- Do not break character or acknowledge that you are a simulation

Response format:
- Respond only with the user's message
- Do not include any meta-commentary or explanations
- Keep responses natural and conversational"""


class UserSimulatorAgentConfig(AgentConfig):
    """Configuration for UserSimulatorAgent."""

    system_prompt: str | None = None

    def __init__(self, system_prompt: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.system_prompt = system_prompt


class UserSimulatorAgent(BaseAgent):
    """Agent that simulates user behavior.

    Uses a persona and scenario to generate realistic user messages
    in a conversation with the agent being tested.
    """

    def __init__(
        self, config: UserSimulatorAgentConfig | AgentConfig | None = None
    ) -> None:
        super().__init__(config)
        self._custom_system_prompt: str | None = None
        if isinstance(config, UserSimulatorAgentConfig):
            self._custom_system_prompt = config.system_prompt

    @property
    def name(self) -> str:
        return "UserSimulator"

    @property
    def system_prompt(self) -> str:
        if self._custom_system_prompt:
            return f"{DEFAULT_USER_SIMULATOR_PROMPT}\n\n---\n\n{self._custom_system_prompt}"
        return DEFAULT_USER_SIMULATOR_PROMPT

    async def generate_first_message(
        self, messages: list[ChatMessage] | None = None
    ) -> str:
        """Generate the first message to start a conversation."""
        prompt_messages = list(messages or [])
        prompt_messages.append(
            ChatMessage(
                role="user",
                content="Generate your first message to start the conversation. Remember your goal and persona.",
            )
        )
        return await self.respond_async(prompt_messages, temperature=0.8)

    def update_context(
        self,
        persona_context: str | None = None,
        scenario_context: str | None = None,
    ) -> None:
        """Update the persona and scenario context."""
        combined = ""
        if persona_context:
            combined += f"PERSONA:\n{persona_context}\n\n"
        if scenario_context:
            combined += f"SCENARIO:\n{scenario_context}"

        self._custom_system_prompt = combined.strip() or None
