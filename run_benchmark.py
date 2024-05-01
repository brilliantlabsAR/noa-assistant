#
# run_benchmark.py
# 
# Benchmark queries for AI assistant. Used for testing and assessing the quality of assistant
# responses. This script talks to a production endpoint and not the Python assistant server
# directly. Simply run:
#
#   python run_benchmark.py tests/tests.json
#
# Use --help for more instructions.
#

import argparse
from datetime import datetime
from enum import Enum
import json
import os
import requests
from typing import List, Optional

from pydantic import BaseModel, RootModel

from models import Capability, MultimodalResponse


####################################################################################################
# Test Case JSON and Evaluation
####################################################################################################

class UserMessage(BaseModel):
    text: str
    image: Optional[str] = None
    capabilities: Optional[List[Capability]] = None # capabilities that are required to have been used
    capabilities_any: Optional[List[Capability]] = None # must use at least one of the capabilities listed here

class TestCase(BaseModel):
    name: str
    active: bool
    default_image: Optional[str] = None
    conversations: List[List[UserMessage | str]]

class TestCaseFile(RootModel):
    root: List[TestCase]

class TestResult(str, Enum):
    FAILED = "FAILED"
    IGNORED = "IGNORED"
    PASSED = "PASSED"

def load_tests(filepath: str) -> List[TestCase]:
    with open(file=filepath, mode="r") as fp:
        text = fp.read()
    return TestCaseFile.model_validate_json(json_data=text).root

def evaluate_capabilities_used(input: UserMessage, output: MultimodalResponse) -> TestResult:
    # Do we have anything to evaluate against?
    has_required_capabilities = input.capabilities is not None and len(input.capabilities) > 0
    has_any_capabilities = input.capabilities_any is not None and len(input.capabilities_any) > 0
    if (not has_required_capabilities) and (not has_any_capabilities):
        # Ignore if desired test results are not specified
        return TestResult.IGNORED
    
    capabilities_used = output.capabilities_used
    
    # Evaluate result against required capabilities
    if has_required_capabilities:
        for required_capability in input.capabilities:
            if required_capability not in capabilities_used:
                return TestResult.FAILED
    
    # Evaluate result against "any capabilities" (an OR function)
    if has_any_capabilities:
        any_present = False
        for interchangeable_capability in input.capabilities_any:
            if interchangeable_capability in capabilities_used:
                any_present = True
        if not any_present:
            return TestResult.FAILED
    
    return TestResult.PASSED


####################################################################################################
# Helper Functions
####################################################################################################

def load_binary_file(filepath: str) -> bytes:
    with open(file=filepath, mode="rb") as fp:
        return fp.read()


####################################################################################################
# Markdown Report Generation
####################################################################################################

class ReportGenerator:
    def __init__(self, test_filepath: str, generate_markdown: bool):
        self._generate_markdown = generate_markdown
        if not generate_markdown:
            return
        base = os.path.splitext(os.path.basename(test_filepath))[0]
        filename = f"{base}.md"
        self._fp = open(file=filename, mode="w")
        self._fp.write(f"# {test_filepath}\n\n")
    
    def __del__(self):
        if not self._generate_markdown:
            return
        self._fp.close()

    def begin_test(self, name: str):
        if not self._generate_markdown:
            return
        self._fp.write(f"## {name}\n\n")
        self._fp.write(f"|Passed?|User|Assistant|Image|Debug|\n")
        self._fp.write(f"|-------|----|---------|-----|-----|\n")

    def begin_conversation(self):
        if not self._generate_markdown:
            return
        self._fp.write("|\\-\\-\\-\\-\\-\\-\\-\\-|\\-\\-\\-\\-\\-\\-\\-\\-|\\-\\-\\-\\-\\-\\-\\-\\-|\\-\\-\\-\\-\\-\\-\\-\\-|\\-\\-\\-\\-\\-\\-\\-\\-|\n")

    def end_conversation(self):
        pass

    def add_result(self, user_message: UserMessage, response: MultimodalResponse, assistant_response: str, test_result: TestResult):
        if not self._generate_markdown:
            return
        passed_column = f"{test_result.value}"
        user_column = self._escape(user_message.text)
        assistant_column = self._escape(assistant_response)
        image_column = f"<img src=\"{user_message.image}\" alt=\"image\" style=\"width:200px;\"/>" if user_message.image is not None else ""
        debug_column = f"```{response.debug_tools}```"
        self._fp.write(f"|{passed_column}|{user_column}|{assistant_column}|{image_column}|{debug_column}|\n")

    def end_test(self, num_passed: int, num_evaluated: int):
        if not self._generate_markdown:
            return
        self._fp.write(f"**Score: {100.0 * num_passed / num_evaluated : .1f}%**\n\n")

    @staticmethod
    def _escape(text: str) -> str:
        special_chars = "\\`'\"*_{}[]()#+-.!"
        escaped_text = ''.join(['\\' + char if char in special_chars else char for char in text])
        return escaped_text.replace("\n", " ")


####################################################################################################
# Main Program
####################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_benchmark")
    parser.add_argument("file", nargs=1)
    parser.add_argument("--endpoint", action="store", default="https://api.brilliant.xyz/dev/noa/mm", help="Address to send request to (Noa server)")
    parser.add_argument("--token", action="store", help="Noa API token")
    parser.add_argument("--test", metavar="name", help="Run specific test")
    parser.add_argument("--markdown", action="store_true", help="Produce report in markdown file")
    parser.add_argument("--vision", action="store", help="Vision model to use (gpt-4-vision-preview, claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-20240229)", default="claude-3-haiku-20240307")
    parser.add_argument("--address", action="store", default="San Francisco, CA 94115", help="Simulated location")
    options = parser.parse_args()

    # Load tests
    tests = load_tests(filepath=options.file[0])

    # Markdown report generator
    report = ReportGenerator(test_filepath=options.file[0], generate_markdown=options.markdown)

    # Authorization header
    headers = {
        "Authorization": options.token if options.token is not None else os.getenv("BRILLIANT_API_KEY")
    }

    # Metrics
    total_user_prompts = 0
    total_tokens_in = 0
    total_tokens_out = 0
    localhost = options.endpoint == "localhost"

    # Run all active tests
    for test in tests:
        if not options.test:
            # No specific test, run all that are active
            if not test.active:
                continue
        else:
            if test.name.lower().strip() != options.test.lower().strip():
                continue

        print(f"Test: {test.name}")
        report.begin_test(name=test.name)
        num_evaluated = 0
        num_passed = 0

        for conversation in test.conversations:
            report.begin_conversation()

            # Create new message history for each conversation
            history = []
            for user_message in conversation:
                # Each user message can be either a string or a UserMessage object
                assert isinstance(user_message, str) or isinstance(user_message, UserMessage)
                if isinstance(user_message, str):
                    user_message = UserMessage(text=user_message)

                # If there is no image associated with this message, use the default image, if it
                # exists
                if user_message.image is None and test.default_image is not None:
                    user_message = user_message.model_copy()
                    user_message.image = test.default_image

                # Construct API call data
                if localhost:
                    options.endpoint = "http://localhost:8000/mm"
                    data = { 
                        "mm": json.dumps({
                                    "prompt": user_message.text,
                                    "messages": history,
                                    "address": options.address,
                                    "local_time": datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),
                                    "config": { "search_api": "serp", "engine": "google_lens" },
                                    "experiment": "1",
                                    "vision": options.vision
                                }
                            ),
                    }
                else:
                    data = { 
                        "prompt": user_message.text,
                        "messages": json.dumps(history),
                        "address": options.address,
                        "local_time": datetime.now().strftime("%A, %B %d, %Y, %I:%M %p"),
                        "config": json.dumps({ "search_api": "serp", "engine": "google_lens" }),
                        "experiment": "1",  # this activates the passthrough to the Python ai-experiments code
                        "vision": options.vision
                    }
                files = {}
                if user_message.image is not None:
                    files["image"] = (os.path.basename(user_message.image), load_binary_file(filepath=user_message.image))

                # Make the call and evaluate
                response = requests.post(url=options.endpoint, files=files, data=data, headers=headers)
                error = False
                try:
                    if response.status_code != 200:
                        print(f"Error: {response.status_code}")
                        print(response.content)
                        response.raise_for_status()
                    #print(response.content)
                    mm_response = MultimodalResponse.model_validate_json(json_data=response.content)
                    #print("Sent:")
                    #print(json.dumps(history))

                    test_result = evaluate_capabilities_used(input=user_message, output=mm_response)
                    if test_result != TestResult.IGNORED:
                        num_evaluated += 1
                        num_passed += (1 if test_result == TestResult.PASSED else 0)

                    history.append({ "role": "user", "content": user_message.text })

                    assistant_response = ""
                    if len(mm_response.response) > 0:
                        assistant_response = mm_response.response
                    elif len(mm_response.image) > 0:
                        assistant_response = "<generated image>"
                    if len(assistant_response) > 0:
                        history.append({ "role": "assistant", "content": assistant_response })

                    print(f"User: {user_message.text}" + (f" ({user_message.image})" if user_message.image else ""))
                    print(f"Response: {assistant_response}")
                    print(f"Tools: {mm_response.debug_tools}")
                    #pct_out = float(content["output_tokens"]) / float(content["total_tokens"]) * 100.0
                    #print(f"Tokens: in={content['input_tokens']}, out={content['output_tokens']} %out={pct_out:.0f}%")
                    print(f"Test: {test_result}")
                    print("")
                    report.add_result(user_message=user_message, response=mm_response, assistant_response=assistant_response, test_result=test_result)

                    total_user_prompts += 1
                    total_tokens_in += mm_response.input_tokens
                    total_tokens_out += mm_response.output_tokens
                    
                except Exception as e:
                    print(f"Error: {e}")

            report.end_conversation()

        # Print test results
        print("")
        print(f"TEST RESULTS: {test.name}")
        print(f"  Score: {num_passed}/{num_evaluated} = {100.0 * num_passed / num_evaluated : .1f}%")
        report.end_test(num_passed=num_passed, num_evaluated=num_evaluated)

    # Summary
    print(f"User messages: {total_user_prompts}")
    print(f"Total input tokens: {total_tokens_in}")
    print(f"Total output tokens: {total_tokens_out}")
    print(f"Average input tokens: {total_tokens_in / total_user_prompts}")
    print(f"Average output tokens: {total_tokens_out / total_user_prompts}")