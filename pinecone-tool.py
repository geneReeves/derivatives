"""
title: Pinecone Tool
description: This tool defines an `EventEmitter` class and a `Tools` class with a `Valves` configuration to query a Pinecone vector database using OpenAI embeddings, returning relevant documents in JSON format.
author: Claude & mike@theelectricrambler.com
version: 0.1.2
last_updated: 19-NOV-2024
license: MIT
requirements: pinecone
"""

import json
import asyncio
import pinecone as Pinecone
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Callable, Any, List, Dict, Union
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class EventEmitter:
    """
    EventEmitter class to handle event emissions.
    """

    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        """
        Initialize the EventEmitter with an optional event emitter function.
        :param event_emitter: A callable that takes a dictionary and returns any type.
        """
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        """
        Emit an event with the given description, status, and done flag.
        :param description: Description of the event.
        :param status: Status of the event.
        :param done: Flag indicating if the event is done.
        """
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class Tools:
    """
    Tools class to handle Pinecone vector database queries using OpenAI embeddings.
    """

    class Valves(BaseModel):
        """
        Configuration class for the Tools class.
        """

        MY_API_KEY: str = Field(
            default="Ollama",
            description="Your OpenAI API key. Ollama is the default.",
        )
        MY_BASE_URL: str = Field(
            default="http://host.containers.internal:11434/v1",
            description="OpenAI, Openrouter, Docker, Podman, Local. Podman is the default.",
            enum=[
                "http://https://api.openai.com/v1",
                "http://host.docker.internal:11434/v1",
                "http://host.containers.internal:11434/v1",
                "http://localhost:11434/v1",
            ],
        )
        PINECONE_API_KEY: str = Field(default="", description="Your Pinecone API key")
        PINECONE_INDEX_NAME: str = Field(
            default="", description="The name of your Pinecone index"
        )
        EMBEDDING_MODEL: str = Field(
            default="mxbai-embed-large",
            description="Embedding model to use. Use text-embedding-ada-002 for OpenAI. mxbai-embed-large is the default for Ollama.",
        )
        EMBEDDING_DIMENSION: int = Field(
            default=1024, description="Dimension of the embedding model output"
        )
        TOP_K: int = Field(default=3, description="Number of top results to return")
        MAX_RETRIES: int = Field(
            default=3,
            description="Maximum number of retries for Pinecone queries",
        )
        RETRY_DELAY: int = Field(
            default=15, description="Delay in seconds between retries"
        )
        CUSTOM_NAME: str = Field(default="Pinecone", description="Tool name here")
        DEBUG: bool = Field(default=False, description="Enable debug.")

    def __init__(self):
        """
        Initialize the Tools class with default Valves configuration.
        """
        self.valves = self.Valves()

    async def query_pinecone(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> Dict[str, Union[List[Dict], str]]:
        """
        Query the Pinecone vector database using the set embeddings and return relevant documents.
        :param query: The query string to search for in the vector database.
        :param __event_emitter__: An optional event emitter function.
        :return: A dictionary containing the query results or error message.
        """
        emitter = EventEmitter(__event_emitter__)
        try:
            # Initial setup progress
            await emitter.emit(description="Starting Pinecone query process...")
            logger.debug("Starting Pinecone query process")

            # OpenAI client setup
            await emitter.emit(description="Setting up client...")
            openai_client = OpenAI(
                base_url=self.valves.MY_BASE_URL, api_key=self.valves.MY_API_KEY
            )
            logger.debug(f"Client initialized with base URL: {self.valves.MY_BASE_URL}")

            # Embedding generation progress
            await emitter.emit(
                description=f"Generating embeddings using {self.valves.EMBEDDING_MODEL}..."
            )
            res = openai_client.embeddings.create(
                model=self.valves.EMBEDDING_MODEL, input=[query]
            )
            xq = res.data[0].embedding
            await emitter.emit(description="Embeddings generated successfully")

            # Dimension verification
            await emitter.emit(description="Verifying embedding dimensions...")
            if len(xq) != self.valves.EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Embedding dimension mismatch. Expected {self.valves.EMBEDDING_DIMENSION}, got {len(xq)}"
                )
            await emitter.emit(description="Embedding dimensions verified")

            # Pinecone connection
            await emitter.emit(
                description=f"Establishing connection to {self.valves.CUSTOM_NAME}..."
            )
            pc = Pinecone(api_key=self.valves.PINECONE_API_KEY)
            index = pc.Index(self.valves.PINECONE_INDEX_NAME)
            await emitter.emit(
                description=f"Successfully connected to {self.valves.CUSTOM_NAME}"
            )

            # Query execution
            contexts = []
            retries = 0

            while (
                len(contexts) < self.valves.TOP_K and retries < self.valves.MAX_RETRIES
            ):
                await emitter.emit(
                    description=f"Executing query attempt {retries + 1}/{self.valves.MAX_RETRIES}..."
                )

                res = index.query(
                    vector=xq,
                    top_k=self.valves.TOP_K,
                    include_metadata=True,
                    namespace=(
                        self.valves.PINECONE_NAMESPACE
                        if hasattr(self.valves, "PINECONE_NAMESPACE")
                        else None
                    ),
                )

                await emitter.emit(
                    description=f"Query attempt {retries + 1} completed. Processing results..."
                )

                contexts = [
                    x.metadata["text"] for x in res.matches if "text" in x.metadata
                ]

                await emitter.emit(
                    description=f"Found {len(contexts)} matching documents"
                )

                if len(contexts) < self.valves.TOP_K:
                    retries += 1
                    if retries < self.valves.MAX_RETRIES:
                        await emitter.emit(
                            description=f"Insufficient results ({len(contexts)}/{self.valves.TOP_K}). Retrying in {self.valves.RETRY_DELAY} seconds..."
                        )
                        await asyncio.sleep(self.valves.RETRY_DELAY)

            # Process results
            await emitter.emit(description="Processing final results...")
            results = []
            for i, context in enumerate(contexts):
                result = {
                    "id": f"result_{i+1}",
                    "content": context,
                    "metadata": res.matches[i].metadata if i < len(res.matches) else {},
                }
                results.append(result)

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [context],
                                "metadata": [
                                    {
                                        "source": result["metadata"].get(
                                            "source", "Unknown source"
                                        )
                                    }
                                ],
                                "source": {
                                    "name": result["metadata"].get(
                                        "title", "Retrieved Sources"
                                    )
                                },
                            },
                        }
                    )
                await emitter.emit(
                    description=f"Processed result {i+1}/{len(contexts)}"
                )

            await emitter.emit(
                status="complete",
                description=f"{self.valves.CUSTOM_NAME} query completed successfully. Retrieved {len(results)} relevant documents.",
                done=True,
            )

            return {"results": results}

        except Exception as e:
            error_msg = str(e)
            await emitter.emit(
                status="error",
                description=f"Error during {self.valves.CUSTOM_NAME} query: {error_msg}",
                done=True,
            )
            logger.error(f"Error during query: {error_msg}")
            return {"error": error_msg}
