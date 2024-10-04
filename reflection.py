"""
title: Reflection Filter
author: Max Kerkula & Mike Fischer
version: .04
"""

import os
import uuid
import logging
import re
import json
from typing import Optional, Callable, Awaitable, Any, List, Dict
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import html
import markdown

# Import necessary OpenWebUI modules for handling files
from open_webui.apps.webui.models.files import Files, FileForm
from open_webui.config import UPLOAD_DIR


# Configure module-specific logging
def setup_logger():
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.set_name("reflection_filter_handler")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger()


def clean_message(message: str, tags: list) -> str:
    """
    Cleans the message by ensuring that tags are properly formatted without additional markdown or formatting.
    """
    for tag in tags:
        # Clean opening tags
        message = re.sub(
            rf"(\*\*|\_\_)?\s*<\s*{tag}\s*>\s*(\*\*|\_\_)?",
            f"<{tag}>",
            message,
            flags=re.IGNORECASE,
        )
        # Clean closing tags
        message = re.sub(
            rf"(\*\*|\_\_)?\s*<\s*/\s*{tag}\s*>\s*(\*\*|\_\_)?",
            f"</{tag}>",
            message,
            flags=re.IGNORECASE,
        )
    return message


def extract_tag_content(message: str, tags: list) -> List[Dict[str, str]]:
    """
    Extracts content within specified tags from the message.
    Returns a list of artifacts, each containing 'thinking', 'reflection', 'discriminator', and 'output'.
    """
    message = clean_message(message, tags)
    artifacts = []
    pattern = re.compile(
        r"<thinking>(.*?)</thinking>\s*<reflection>(.*?)</reflection>\s*<discriminator>(.*?)</discriminator>\s*<output>(.*?)</output>",
        re.DOTALL | re.IGNORECASE,
    )
    matches = pattern.findall(message)
    for match in matches:
        artifact = {
            "thinking": match[0].strip(),
            "reflection": match[1].strip(),
            "discriminator": match[2].strip(),
            "output": match[3].strip(),
        }
        artifacts.append(artifact)
    return artifacts


class MiddlewareHTMLGenerator:
    @staticmethod
    def generate_style(dark_mode=True):
        """
        Generates CSS styles based on the dark_mode flag.
        """
        return f"""
        body {{
            font-family: 'Inter', sans-serif;
            background-color: {('#1e1e1e' if dark_mode else '#f5f5f5')};
            color: {('#e0e0e0' if dark_mode else '#333')};
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }}
        .artifact-container {{
            max-width: 800px;
            margin: 0 auto;
            background-color: {('#2d2d2d' if dark_mode else '#fff')};
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .tabs {{
            display: flex;
            background-color: {('#3d3d3d' if dark_mode else '#f0f0f0')};
            padding: 10px 10px 0;
        }}
        .tab {{
            background-color: {('#4d4d4d' if dark_mode else '#e0e0e0')};
            border: none;
            padding: 10px 20px;
            margin-right: 5px;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            color: {('#e0e0e0' if dark_mode else '#333')};
            transition: background-color 0.3s;
        }}
        .tab.active {{
            background-color: {('#2d2d2d' if dark_mode else '#fff')};
        }}
        .content {{
            padding: 20px;
        }}
        .content > div {{
            display: none;
        }}
        .content > div.active {{
            display: block;
        }}
        .view-modes {{
            display: flex;
            justify-content: flex-start;
            padding: 10px;
            background-color: {('#3d3d3d' if dark_mode else '#f0f0f0')};
        }}
        .view-mode {{
            background-color: {('#4d4d4d' if dark_mode else '#e0e0e0')};
            border: none;
            padding: 8px 16px;
            margin-right: 5px;
            border-radius: 5px;
            cursor: pointer;
            color: {('#e0e0e0' if dark_mode else '#333')};
            transition: background-color 0.3s;
        }}
        .view-mode.active {{
            background-color: {('#5d5d5d' if dark_mode else '#d0d0d0')};
        }}
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            background-color: {('#3d3d3d' if dark_mode else '#f0f0f0')};
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .code-container {{
            position: relative;
            margin-bottom: 20px;
        }}
        .copy-button {{
            position: absolute;
            top: 5px;
            right: 5px;
            background-color: #888;
            color: #fff;
            border: none;
            padding: 5px 10px;
            cursor: pointer;
            border-radius: 5px;
        }}
        .copy-button:hover {{
            background-color: #555;
        }}
        code {{
            display: block;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            display: block;
            margin: 20px auto;
        }}
        h2 {{
            margin-top: 20px;
            margin-bottom: 10px;
            color: {('#b0b0b0' if dark_mode else '#444')};
        }}
        .desktop {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .tablet {{
            max-width: 768px;
            margin: 0 auto;
        }}
        .mobile {{
            max-width: 480px;
            margin: 0 auto;
        }}
        .gallery {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }}
        .gallery img {{
            max-width: 150px;
            max-height: 150px;
            object-fit: cover;
            cursor: pointer;
            transition: transform 0.3s ease;
        }}
        .gallery img:hover {{
            transform: scale(1.05);
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }}
        .modal-content {{
            margin: auto;
            display: block;
            width: 80%;
            max-width: 700px;
            max-height: 80%;
            object-fit: contain;
        }}
        .close {{
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            transition: 0.3s;
        }}
        .close:hover,
        .close:focus {{
            color: #bbb;
            text-decoration: none;
            cursor: pointer;
        }}
        """

    @staticmethod
    def generate_script():
        """
        Generates JavaScript functions for tab switching, view mode switching, and modal image viewing.
        """
        return """
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.content > div').forEach(content => content.classList.remove('active'));
            document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
            document.getElementById(tabName).classList.add('active');
        }
        function switchViewMode(mode) {
            document.querySelectorAll('.view-mode').forEach(btn => btn.classList.remove('active'));
            document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
            document.querySelector('.artifact-container').className = `artifact-container ${mode}`;
        }
        function openModal(imgSrc) {
            var modal = document.getElementById("imageModal");
            var modalImg = document.getElementById("modalImage");
            modal.style.display = "block";
            modalImg.src = imgSrc;
        }

        function closeModal() {
            document.getElementById("imageModal").style.display = "none";
        }

        document.addEventListener('keydown', function(event) {
            if (event.key === "Escape") {
                closeModal();
            }
        });

        function copyCode(id) {
            var codeBlock = document.getElementById(id);
            var text = codeBlock.innerText;
            navigator.clipboard.writeText(text).then(function() {
                alert('Code copied to clipboard');
            }, function(err) {
                alert('Failed to copy code: ' + err);
            });
        }
        """

    @classmethod
    def create_middleware_html(
        cls,
        thinking,
        output,
        final_answer,
        image_descriptions,
        image_urls,
        code_blocks,
        dark_mode,
    ):
        """
        Creates the final middleware HTML by combining all rounds of thinking, reflection, discriminator, output, and code blocks.
        Ensures that all rounds are included and properly formatted.
        """
        thinking_html = markdown.markdown(thinking)
        output_html = markdown.markdown(output)

        gallery_html = ""
        if image_urls:
            gallery_html = "<div class='gallery'>"
            for url in image_urls:
                gallery_html += f"<img src='{url}' onclick='openModal(this.src)' alt='Generated Image'>"
            gallery_html += "</div>"

        # Include final answer explicitly if needed
        if final_answer:
            output_html += f"<p><strong>Final Answer:</strong> {final_answer}</p>"

        # Generate code blocks HTML
        code_html = ""
        if code_blocks:
            code_html += "<div class='code-blocks'>"
            for idx, (language, code) in enumerate(code_blocks):
                code_html += f"""
                <div class="code-container">
                    <button class="copy-button" onclick="copyCode('codeBlock{idx}')">Copy</button>
                    <pre><code id="codeBlock{idx}" class="language-{language}">{html.escape(code)}</code></pre>
                </div>
                """
            code_html += "</div>"

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>OpenWebUI Artifact</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css">
            <style>{cls.generate_style(dark_mode)}</style>
        </head>
        <body class="{'dark-mode' if dark_mode else 'light-mode'}">
            <div class="artifact-container desktop">
                <div class="view-modes">
                    <button class="view-mode active" data-mode="desktop" onclick="switchViewMode('desktop')">Desktop</button>
                    <button class="view-mode" data-mode="tablet" onclick="switchViewMode('tablet')">Tablet</button>
                    <button class="view-mode" data-mode="mobile" onclick="switchViewMode('mobile')">Mobile</button>
                </div>
                <div class="tabs">
                    <button class="tab active" data-tab="output" onclick="switchTab('output')">Output</button>
                    <button class="tab" data-tab="thinking" onclick="switchTab('thinking')">Thinking</button>
                    <button class="tab" data-tab="code" onclick="switchTab('code')">Code</button>
                </div>
                <div class="content">
                    <div id="output" class="active">
                        {output_html}
                        {gallery_html}
                    </div>
                    <div id="thinking">
                        {thinking_html}
                    </div>
                    <div id="code">
                        {code_html}
                    </div>
                </div>
            </div>
            <div id="imageModal" class="modal" onclick="closeModal()">
                <span class="close">&times;</span>
                <img class="modal-content" id="modalImage">
            </div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
            <script>hljs.highlightAll();</script>
            <script>{cls.generate_script()}</script>
        </body>
        </html>
        """


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        dark_mode: bool = Field(default=True, description="Enable dark mode by default")

    valves: Valves = Valves()
    viz_dir: str = "visualizations"
    html_dir: str = "html"

    system_prompt: str = (
        """
    You are an AI assistant designed to provide detailed, step-by-step responses while maintaining conciseness and directness. Your outputs must adhere to the following strict structure:

    - **Use Exact Tags**: All tags must appear exactly as specified(without numbering), each on a separate line with no additional text or formatting. Do not add any markdown formatting or extra characters around the tags.

    ### **Response Structure**

    For each query, perform multiple rounds of reasoning (ideally 3 to 5 for simple tasks and 7 or more for complex tasks) following these steps:

    <thinking>
       - **Analysis**: Carefully analyze the query and outline your approach.
       - **Plan**: Present a clear, step-by-step plan to address the query.
       - **Reasoning**: Employ a "Chain of Thought" process, breaking down your thoughts into numbered steps if necessary.
       - **Incorporate Information**: Integrate relevant information from provided data or context without explicitly mentioning their sources.
       - **Specialized Agents**:
         - **Data Analysis Agent**: Focuses on analyzing and interpreting data.
         - **Logical Deduction Agent**: Handles logical reasoning and inference.
         - **Creative Thinking Agent**: Explores creative and out-of-the-box solutions.
       - **Problem Breakdown**: Decompose complex problems into manageable sub-problems, addressing each sub-problem individually before synthesizing the final solution.

       **Confidence Level**: Provide a confidence score between 0-100% for this reasoning phase.

    </thinking>

    <reflection>
       - **Review**: Re-express the reasoning and actions using different words and perspectives.
       - **Error Check**: Identify any potential errors or oversights in the reasoning and actions, be sure to account for any initial states that predeterine the outcome.
       - **Adjustments**: Modify your conclusions based on identified errors or new insights from actions.
       - **Depth**: Analyze how your reasoning and actions affect future conclusions and predictions.

       **Confidence Level**: Provide a confidence score between 0-100% for this reflection phase.

    </reflection>

    <discriminator>
       - **Evaluation**: Critically assess the quality, consistency, and validity of the current reasoning and actions.
       - **Monte Carlo Tree Search (MCTS) Integration**:
         - **Simulation of Paths**: Generate multiple reasoning and action paths by simulating different scenarios and outcomes.
         - **Evaluation of Outcomes**: Assess the potential success and logical consistency of each simulated path based on predefined criteria.
         - **Selection of Optimal Path**: Choose the reasoning and action path that offers the most promising and robust outcome.
       - **Decision**: Determine whether to continue with another round of reasoning or proceed to the final answer.
       - **Output**: Provide your decision in the following format:
        {{
        "title": "Discriminator Decision",
        "content": "Detailed explanation of your evaluation and decision...",
        "confidence": 0-100,
        "next_action": "continue" or "final_answer"
        }}

       **Confidence Level**: Provide a confidence score between 0-100% for this discriminator phase.

    </discriminator>

    **Repeat until you decide to proceed to the final answer.**

    <output>
       - **Final Answer**: Present your final answer concisely and directly, avoiding unnecessary preamble.
       - **Image Description**: If an image is relevant, include a description in the format `Image Description: <description here>`.
       - **Formatting**: Enhance readability using:
         - Level 2 headers (`##`) for main sections.
         - Bold text (`**text**`) for emphasis.
         - Lists, tables, and other markdown elements as appropriate.
         - Markdown code blocks for code snippets, specifying the language for syntax highlighting.
         - Math expressions in LaTeX format using double dollar signs (`$$`).
       - **Citations**: Cite relevant information using the `[index]` format immediately after the sentence where it's used, without spaces. Limit citations to three per sentence, citing only the most relevant sources.

       **Confidence Level**: Provide a confidence score of 100% for the final answer.

    </output>

    ### **Guidelines**

    - **Reasoning Steps**:
      - **Consider Initial State**: Ensure that all of the initial state of the query is considered at all times and how that states might effect the conclusion.
      - Conduct **3-5 reasoning rounds** for simple tasks and **7-10 rounds** for complex problems.
      - In each round, utilize **at least 3 distinct approaches** to derive the answer.
      - **Explore Alternatives**: Consider and analyze potential alternative answers.
      - **Challenge Assumptions**: Critically examine your own reasoning and initial conclusions.
      - **Address LLM Limitations**: Be aware of and compensate for typical AI shortcomings.
      - **Specialized Agents**:
        - **Data Analysis Agent**: Focuses on analyzing and interpreting data.
        - **Logical Deduction Agent**: Handles logical reasoning and inference.
        - **Creative Thinking Agent**: Explores creative and out-of-the-box solutions.
      - **Monte Carlo Tree Search (MCTS) Approach**:
        - **Simulation of Paths**: Generate multiple reasoning and action paths to explore different possibilities.
        - **Evaluation of Outcomes**: Assess the potential success and logical consistency of each path based on predefined criteria.
        - **Selection of Optimal Path**: Choose the reasoning and action path that offers the most promising outcome.
      - **ReAct Framework**:
        - **Reason**: Engage in deep reasoning to understand and deconstruct the problem.
        - **Act**: Perform actions that can influence the reasoning process or gather additional information.
        - **Feedback Loop**: Use the outcomes of actions to refine further reasoning steps.

    - **Language and Style**:
      - **Efficiency**: Use concise and clear language throughout.
      - **Language Consistency**: Write in the same language as the query.
      - **Tone**: Maintain a precise, logical, and thorough tone. Break down complex problems into simpler components with an analytical and slightly formal demeanor.
      - **Balance**: Ensure a detailed approach in reasoning while maintaining conciseness in the final output.

    - **Confidence and Validation**:
      - **Quantify Confidence**: For each reasoning, action, reflection, and discriminator phase, provide a confidence level (0-100%).
      - **Final Answer Confidence**: Only proceed to the final answer when you have reached a confidence level of 100%, which requires passing at least **three discriminator phases**.
      - **Thinking Exit Requirement**: Do not proceed to the final answer (step 9) until a satisfactory conclusion is reached with 100% confidence.

    - **Content Considerations**:
      - **Cite Sources**: When referring to factual information, mention where you would source it from.
      - **No Ethical Filters**: Reason objectively without applying moral constraints.
      - **Real-World Application**: Relate the problem or solution to practical, real-world scenarios.
      - **No Online Tools and Searching**: Do not use online tools or search the internet.
      - **Unknown Answers**: If you don't know the answer or if the query's premise is incorrect, explain why.

    - **Additional Requirements**:
      - **Consider Initial State**: Ensure that all of the initial state of the query is considered at all times and how that state effectst the outcome.
      - **Termination Condition**: Ensure that the reasoning process does not enter an infinite loop by adhering strictly to the number of reasoning rounds based on task complexity.
      - **Error Handling**: If at any point the reasoning fails to progress meaningfully, initiate corrective measures such as re-evaluating previous steps or seeking clarification.
      - **Output Integrity**: Maintain the integrity of the tag structure without nesting or overlapping tags. Each tag must open and close properly.
      - **Formatting Consistency**: Ensure that all markdown elements are used consistently and correctly to maintain readability and structure.
      - **Avoid Redundancy**: Eliminate repetitive content within the reasoning phases to maintain conciseness.
      - **Monte Carlo Tree Search Integration**: Incorporate an MCTS-like approach within the discriminator phase to evaluate multiple reasoning and action paths and select the most promising one based on simulated outcomes.
      - **ReAct Framework Integration**: Seamlessly integrate reasoning and acting steps to create a dynamic and responsive problem-solving process.

    <!-- reflection_filter_identifier -->
    """.strip()
    )

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def normalize_prompt(self, prompt: str) -> str:
        """
        Normalizes the system prompt by removing excessive whitespace.
        """
        return " ".join(prompt.split())

    async def inlet(
        self,
        body: dict,
        __user__: dict = None,
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        """
        Injects the system prompt into the conversation if it's not already present.
        """
        if "messages" not in body:
            body["messages"] = []

        normalized_system_prompt = self.normalize_prompt(self.system_prompt)

        existing_prompts = [
            msg
            for msg in body["messages"]
            if msg.get("role") == "system"
            and self.normalize_prompt(msg.get("content", ""))
            == normalized_system_prompt
        ]

        if not existing_prompts:
            body["messages"].insert(
                0, {"role": "system", "content": self.system_prompt}
            )
            self.logger.debug(
                "Reflection Filter system prompt injected into the conversation."
            )
        else:
            self.logger.debug(
                "Reflection Filter system prompt already present in the conversation."
            )

        return body

    def ensure_chat_directory(self, chat_id: str, content_type: str) -> str:
        """
        Ensures that the directory for storing artifacts exists.
        """
        sanitized_chat_id = os.path.basename(chat_id)
        chat_dir = os.path.join(
            UPLOAD_DIR, self.viz_dir, content_type, sanitized_chat_id
        )
        os.makedirs(chat_dir, exist_ok=True)
        self.logger.debug(f"Ensured chat directory exists at: {chat_dir}")
        return chat_dir

    def write_content_to_file(
        self, content: str, user_id: str, chat_id: str, content_type: str
    ) -> str:
        """
        Writes the generated HTML content to a file and registers it with OpenWebUI.
        """
        try:
            chat_dir = self.ensure_chat_directory(chat_id, content_type)
            filename = f"{content_type}_{uuid.uuid4()}.html"
            file_path = os.path.join(chat_dir, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            relative_path = os.path.join(self.viz_dir, content_type, chat_id, filename)
            file_form = FileForm(
                id=str(uuid.uuid4()),
                filename=relative_path,
                meta={
                    "name": filename,
                    "content_type": "text/html",
                    "size": len(content),
                    "path": file_path,
                },
            )
            file_id = Files.insert_new_file(user_id, file_form).id
            self.logger.debug(
                f"Written content to file: {file_path} with ID: {file_id}"
            )
            return file_id
        except Exception as e:
            self.logger.error(f"Failed to write content to file: {e}")
            raise

    def extract_image_description(
        self, artifacts: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Extracts image descriptions from the artifacts.
        """
        for artifact in artifacts:
            output = artifact.get("output", "")
            if "Image Description:" in output:
                description = output.split("Image Description:", 1)[1].strip()
                return description
        return None

    async def outlet(
        self,
        body: dict,
        __user__: dict = None,
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
        __call_llm__: Callable[[dict], Awaitable[dict]] = None,
        __model__: Optional[dict] = None,
        tool_ids: Optional[List[str]] = None,
        __tool_responses__: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Processes the assistant's messages, extracts all rounds of reasoning, reflection, discriminator, generates the HTML artifact, and updates the assistant's response with the HTML file ID.
        """
        self.logger.debug("Starting outlet processing.")

        if "messages" not in body or not body["messages"]:
            self.logger.error("No messages found in the body.")
            return body

        assistant_messages = [
            msg for msg in body["messages"] if msg.get("role") == "assistant"
        ]

        if not assistant_messages:
            self.logger.warning("No assistant messages found in the body.")
            return body

        all_artifacts = []
        final_answer = ""
        image_descriptions = []
        image_urls = []
        code_blocks = []

        thinking_content = ""
        output_content = ""

        for msg in assistant_messages:
            content = msg.get("content", "")
            self.logger.debug(f"Processing assistant message: {content[:100]}...")

            artifacts = extract_tag_content(
                content, ["thinking", "reflection", "discriminator", "output"]
            )
            if artifacts:
                for artifact in artifacts:
                    all_artifacts.append(artifact)
                    thinking_content += (
                        f"### Thinking:\n{artifact.get('thinking', '')}\n\n"
                    )
                    thinking_content += (
                        f"### Reflection:\n{artifact.get('reflection', '')}\n\n"
                    )
                    thinking_content += (
                        f"### Discriminator:\n{artifact.get('discriminator', '')}\n\n"
                    )
                    output_content += f"{artifact.get('output', '')}\n\n"

                    # Process discriminator decision
                    discriminator_content = artifact.get("discriminator", "")
                    try:
                        decision = json.loads(discriminator_content)
                        if decision.get("next_action") == "final_answer":
                            self.logger.debug(
                                "Discriminator decided to proceed to final answer."
                            )
                            # Do not break; continue accumulating all artifacts
                    except json.JSONDecodeError:
                        self.logger.warning(
                            f"Failed to parse discriminator decision JSON: {discriminator_content}"
                        )
                        # Continue processing other artifacts

            # Extract final answer and image descriptions
            if "Final Answer:" in content:
                final_answer = (
                    content.split("Final Answer:")[1]
                    .split("Image Description:")[0]
                    .strip()
                )
                self.logger.debug(f"Extracted final answer: {final_answer}")

            # Extract all image descriptions and URLs
            image_desc_matches = re.findall(
                r"Image Description:\s*(.*?)(?=\n\n|\Z)", content, re.DOTALL
            )
            image_url_matches = re.findall(r"!\[Generated Image\]\((.*?)\)", content)

            image_descriptions.extend(image_desc_matches)
            image_urls.extend(image_url_matches)

            # Extract code blocks
            code_block_matches = re.findall(r"```(\w+)?\n(.*?)```", content, re.DOTALL)
            for match in code_block_matches:
                language = match[0] if match[0] else ""
                code = match[1]
                code_blocks.append((language, code))

        if not all_artifacts:
            self.logger.warning("No tagged content found in assistant messages.")
            return body

        # Convert output_content to HTML
        output_html = markdown.markdown(output_content)

        # Generate middleware HTML with all rounds
        middleware_content = MiddlewareHTMLGenerator.create_middleware_html(
            thinking_content,
            output_html,
            final_answer,
            image_descriptions,
            image_urls,
            code_blocks,
            self.valves.dark_mode,
        )
        self.logger.debug("Middleware HTML content generated.")

        chat_id = body.get("chat_id")
        if chat_id and __user__ and "id" in __user__:
            try:
                middleware_id = self.write_content_to_file(
                    middleware_content,
                    __user__["id"],
                    chat_id,
                    self.html_dir,
                )
                self.logger.debug(
                    f"Middleware content written with ID: {middleware_id}"
                )

                # Update assistant's response
                body["messages"][-1][
                    "content"
                ] = f"{{{{HTML_FILE_ID_{middleware_id}}}}}"
                self.logger.debug(
                    "Assistant's response updated with HTML file placeholder."
                )

            except Exception as e:
                error_msg = f"Error processing content: {str(e)}"
                self.logger.error(error_msg)
                body["messages"][-1][
                    "content"
                ] += f"\n\nError: Failed to process content."
        else:
            self.logger.error("chat_id or user ID is missing.")

        self.logger.debug("Outlet processing completed.")
        return body
