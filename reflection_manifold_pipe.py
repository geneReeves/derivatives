"""
title: Reflection Manifold Pipe Updated
author: flomanxl | AndreyRGW | mkfischer
version: 1.4
required_open_webui_version: 0.3.30
"""

from pydantic import BaseModel, Field
from typing import Optional, Callable, Awaitable
import aiohttp
import logging
import traceback


class Action:
    class Valves(BaseModel):
        model: str = Field(
            default="qwen2.5:32b-instruct",
            description="Model to use for reflection process.",
        )
        api_base: str = Field(
            default="http://host.containers.internal:11434/v1",
            description="Base URL for the model API.",
        )
        enable_status_indicator: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )
        debug_mode: bool = Field(
            default=False, description="Enable or disable debug mode"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if self.valves.debug_mode else logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    async def action(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> Optional[dict]:
        self.logger.info("Starting reflection process")
        await self.emit_status(
            __event_emitter__, "info", "Starting reflection process", False
        )

        try:
            messages = body.get("messages", [])
            if not messages:
                raise ValueError("No messages found in the request body")

            self.logger.debug(f"Processing {len(messages)} messages")
            await self.emit_progress(__event_emitter__, 10)

            initial_response = await self.process_thinking(
                messages[-1]["content"], __event_emitter__
            )
            self.logger.debug(f"Initial response: {initial_response}")
            await self.emit_progress(__event_emitter__, 40)

            reflection_response = await self.process_reflection(
                initial_response, __event_emitter__
            )
            self.logger.debug(f"Reflection response: {reflection_response}")
            await self.emit_progress(__event_emitter__, 70)

            final_response = await self.process_output(
                reflection_response, __event_emitter__
            )
            self.logger.debug(f"Final response: {final_response}")
            await self.emit_progress(__event_emitter__, 90)

            if not final_response:
                raise ValueError("Final response is empty")

            if messages[-1]["role"] == "assistant":
                messages[-1]["content"] += f"\n\nReflection output:\n{final_response}"
            else:
                body["messages"].append(
                    {"role": "assistant", "content": final_response}
                )

            self.logger.info("Reflection process completed successfully")
            await self.emit_status(
                __event_emitter__, "info", "Reflection process completed", True
            )
            await self.emit_progress(__event_emitter__, 100)

            return body

        except Exception as e:
            error_msg = f"Error in reflection process: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            await self.emit_status(__event_emitter__, "error", error_msg, True)
            return {"error": error_msg}

    async def process_thinking(
        self, prompt: str, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> str:
        self.logger.info("Generating initial thinking")
        await self.emit_status(
            __event_emitter__, "info", "Generating initial thinking", False
        )
        response = await self.query_model(
            f"<thinking>{prompt}</thinking>", __event_emitter__
        )
        return response

    async def process_reflection(
        self,
        thinking_response: str,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        self.logger.info("Processing reflection")
        await self.emit_status(
            __event_emitter__, "info", "Processing reflection", False
        )
        reflection_prompt = f"<reflection>{thinking_response}</reflection>"
        reflection_response = await self.query_model(
            reflection_prompt, __event_emitter__
        )
        return reflection_response

    async def process_output(
        self,
        reflection_response: str,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        self.logger.info("Generating final output")
        await self.emit_status(
            __event_emitter__, "info", "Generating final output", False
        )
        final_output = await self.query_model(
            f"<output>{reflection_response}</output>", __event_emitter__
        )
        return final_output

    async def query_model(
        self, prompt: str, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> str:
        url = f"{self.valves.api_base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.valves.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            self.logger.info(f"Querying model: {self.valves.model}")
            await self.emit_status(
                __event_emitter__, "info", f"Querying model: {self.valves.model}", False
            )
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        error_msg = f"API request failed with status {response.status}"
                        self.logger.error(error_msg)
                        return f"Error: {error_msg}"
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
        except Exception as e:
            error_msg = f"Error querying model: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            return f"Error: {error_msg}"

    async def emit_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        level: str,
        message: str,
        done: bool,
    ):
        if __event_emitter__ and self.valves.enable_status_indicator:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )

    async def emit_progress(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        progress: int,
    ):
        if __event_emitter__ and self.valves.enable_status_indicator:
            await __event_emitter__(
                {
                    "type": "progress",
                    "data": {
                        "progress": progress,
                    },
                }
            )

    async def on_start(self):
        self.logger.info("Reflection Process Action Started")

    async def on_stop(self):
        self.logger.info("Reflection Process Action Stopped")
