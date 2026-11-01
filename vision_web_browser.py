"""
vision_web_browser.py

An AI-powered web automation script using the `smolagents` framework,
Selenium WebDriver, Helium, and a vision-capable Large Language Model (LLM).

The agent navigates websites, takes visual screenshots after every step,
inspects page content, and solves complex web browsing tasks autonomously.
"""

# ==============================================================================
# 1. IMPORTS & DEPENDENCIES
# ==============================================================================

# 'argparse' is a standard Python library module used to parse command-line arguments.
# It allows the user to pass options when running the script from a terminal (e.g., `python script.py --model-id gpt-4o`).
import argparse

# 'BytesIO' from the 'io' module creates an in-memory binary stream (a buffer).
# Instead of saving raw image bytes to a physical hard drive file and reading them back,
# BytesIO treats raw byte data in RAM as if it were an open file.
from io import BytesIO

# 'sleep' pauses program execution for a specified number of seconds.
# Used here to wait for web pages and JavaScript animations to finish rendering.
from time import sleep

# 'helium' is a high-level Python library for web automation built on top of Selenium.
# It provides human-like commands (e.g., click("Submit"), write("hello")) that are simpler than raw Selenium.
import helium

# 'load_dotenv' reads key-value pairs from a local '.env' file and adds them to environment variables (os.environ).
# This is commonly used to safely load secret API keys (like OPENAI_API_KEY) without hardcoding them in scripts.
from dotenv import load_dotenv

# 'Image' from the 'PIL' (Pillow) library provides tools to open, manipulate, inspect, and save images in Python.
from PIL import Image

# 'webdriver' is the core module of Selenium that controls a real web browser (like Google Chrome).
from selenium import webdriver

# 'By' provides standardized locator mechanisms in Selenium (e.g., By.ID, By.XPATH, By.CLASS_NAME) to find HTML elements.
from selenium.webdriver.common.by import By

# 'Keys' contains representations of special non-printable keyboard keys (such as ESCAPE, ENTER, BACKSPACE, TAB).
from selenium.webdriver.common.keys import Keys

# 'CodeAgent' is an autonomous agent from Hugging Face's 'smolagents' that writes and executes Python code to solve tasks.
# 'WebSearchTool' is a prebuilt tool that allows the agent to perform web searches (e.g., via DuckDuckGo / Google).
# 'tool' is a decorator used to convert regular Python functions into tools that the AI agent can call.
from smolagents import CodeAgent, WebSearchTool, tool

# 'ActionStep' is a data structure representing a single action/thought/observation cycle taken by the agent in its memory.
from smolagents.agents import ActionStep

# 'load_model' is a helper function from smolagents to instantiate an LLM client (e.g., LiteLLM, OpenAI, Hugging Face).
from smolagents.cli import load_model


# ==============================================================================
# 2. DEFAULT PROMPT / TASK DEFINITION
# ==============================================================================

# A default multi-line string prompt representing the mission given to the agent.
# Here, the agent acts as Alfred verifying a superhero visitor ("Wonder Woman") using web search and Wikipedia.
alfred_guest_list_request = """
I am Alfred, the butler of Wayne Manor, responsible for verifying the identity of guests at party. A superhero has arrived at the entrance claiming to be Wonder Woman, but I need to confirm if she is who she says she is.
Please search for images of Wonder Woman and generate a detailed visual description based on those images. Additionally, navigate to Wikipedia to gather key details about her appearance. With this information, I can determine whether to grant her access to the event.
"""


# ==============================================================================
# 3. COMMAND-LINE ARGUMENT PARSING
# ==============================================================================

def parse_arguments():
    """
    Sets up and parses terminal command-line arguments for the script.
    
    Returns:
        argparse.Namespace: An object containing all parsed arguments as attributes.
    """
    # Create the parser object with a human-readable description of what the script does.
    parser = argparse.ArgumentParser(
        description="Run a web browser automation script with a specified model."
    )
    
    # Positional argument: "prompt"
    # - type=str: converts input to string.
    # - nargs="?": indicates this argument is optional (can take 0 or 1 value).
    # - default=alfred_guest_list_request: if no prompt is given in the terminal, fall back to the Alfred scenario.
    # - help: descriptive text shown if the user runs `python script.py --help`.
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",  # Makes it optional
        default=alfred_guest_list_request,
        help="The prompt to run with the agent",
    )
    
    # Optional flag: "--model-type"
    # Specifies which backend provider class smolagents will use (default: LiteLLMModel).
    parser.add_argument(
        "--model-type",
        type=str,
        default="LiteLLMModel",
        help="The model type to use (e.g., OpenAIServerModel, LiteLLMModel, TransformersModel, InferenceClientModel)",
    )
    
    # Optional flag: "--model-id"
    # Specifies the specific model identifier/checkpoint name (e.g., "gpt-4o", "claude-3-5-sonnet", etc.).
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt-4o",
        help="The model ID to use for the specified model type",
    )
    
    # Inspects sys.argv, parses the arguments against the rules defined above, and returns the result.
    return parser.parse_args()


# ==============================================================================
# 4. STEP CALLBACK: SCREENSHOT & OBSERVATION CAPTURE
# ==============================================================================

def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    """
    A callback function executed automatically after every step the agent takes.
    It captures a browser screenshot, compresses past memory to save tokens/RAM,
    and attaches the screenshot and current URL to the agent's observation.

    Parameters:
        memory_step (ActionStep): The current step record being added to agent memory.
        agent (CodeAgent): The agent instance running the task.
    """
    # Pause for 1.0 second to allow CSS animations, page transitions, and DOM updates to stabilize.
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    
    # Retrieve the Selenium WebDriver instance currently being controlled by Helium.
    driver = helium.get_driver()
    
    # Retrieve the integer step number of the current action (e.g., step 1, step 2, ...).
    current_step = memory_step.step_number
    
    # Verify that a browser driver actually exists before trying to capture a screenshot.
    if driver is not None:
        # Loop through all past steps recorded in the agent's memory history.
        for previous_memory_step in agent.memory.steps:  # Remove previous screenshots from logs for lean processing
            # Check if the past step is an ActionStep and is older than the immediately preceding step (2 steps back).
            # Why? High-resolution screenshots consume massive amounts of tokens when sent to vision LLMs.
            # Keeping only the most recent screenshot(s) prevents hitting context-window limits and reduces latency & cost.
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                # Discard the old screenshot image data to free up memory and LLM context space.
                previous_memory_step.observations_images = None
        
        # Capture the current browser window as raw binary PNG data (bytes).
        png_bytes = driver.get_screenshot_as_png()
        
        # BytesIO wraps the raw bytes in an in-memory stream, and Image.open parses it into a PIL Image object.
        image = Image.open(BytesIO(png_bytes))
        
        # Log the image dimensions (width, height) to the console for debugging and visibility.
        print(f"Captured a browser screenshot: {image.size} pixels")
        
        # image.copy() creates a deep copy of the image in memory so that subsequent modifications
        # or closing of the underlying buffer does not corrupt the screenshot stored in the agent's memory.
        memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists, important!

    # Retrieve the current URL of the page from the browser driver and format it into a status string.
    url_info = f"Current url: {driver.current_url}"
    
    # Update memory_step.observations:
    # If observations is empty (None), assign url_info directly.
    # Otherwise, append the url_info to the existing observation text on a new line.
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )
    
    # Explicit return statement ending the callback.
    return


# ==============================================================================
# 5. CUSTOM AGENT TOOLS
# ==============================================================================

# The @tool decorator registers this function as a callable tool for the smolagents framework.
# The framework reads the docstring, type annotations, and function signature to teach the LLM how to invoke it.
@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    # By.XPATH is a query language used to search XML/HTML documents.
    # "//*[contains(text(), '{text}')]" searches all elements ('*') anywhere in the DOM ('//')
    # whose text contains the specified substring.
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    
    # Boundary validation: if the caller asks for the 5th match, but only 2 were found, raise an error.
    if nth_result > len(elements):
        raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
    
    # Build a feedback message indicating how many matches were located.
    result = f"Found {len(elements)} matches for '{text}'."
    
    # Python lists are 0-indexed, so the nth occurrence corresponds to index (nth_result - 1).
    elem = elements[nth_result - 1]
    
    # driver.execute_script runs raw JavaScript inside the browser window.
    # "arguments[0].scrollIntoView(true);" instructs the browser to scroll the webpage until
    # the target element ('elem', passed as arguments[0]) aligns to the top of the visible viewport.
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    
    # Append the focused position information to the result string.
    result += f"Focused on element {nth_result} of {len(elements)}"
    
    # Return the message so the AI agent receives confirmation in its observation step.
    return result


@tool
def go_back() -> None:
    """Goes back to previous page."""
    # driver.back() simulates the user clicking the browser's "Back" navigation button.
    driver.back()


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    # ActionChains is a Selenium utility for queuing and executing complex low-level user interactions.
    # Here, it simulates pressing the 'ESCAPE' key (Keys.ESCAPE) on the page to dismiss open dialogs/modals.
    # .perform() fires the queued keystroke action in the browser.
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()


# ==============================================================================
# 6. BROWSER & AGENT INITIALIZATION
# ==============================================================================

def initialize_driver():
    """Initialize the Selenium WebDriver."""
    # ChromeOptions allows configuring Chrome startup flags, extensions, and behaviors.
    chrome_options = webdriver.ChromeOptions()
    
    # Sets device scale factor to 1.0 (100% zoom) to prevent High-DPI / Retina scaling distortions on screenshots.
    chrome_options.add_argument("--force-device-scale-factor=1")
    
    # Defines the browser window's starting resolution: 1000 pixels wide by 1350 pixels high.
    chrome_options.add_argument("--window-size=1000,1350")
    
    # Disables the built-in Chrome PDF viewer extension so PDFs can be downloaded or handled directly.
    chrome_options.add_argument("--disable-pdf-viewer")
    
    # Places the browser window at coordinates (x=0, y=0), the top-left corner of the primary display.
    chrome_options.add_argument("--window-position=0,0")
    
    # helium.start_chrome opens a visible (headless=False) Chrome instance applying the custom options.
    # It also binds Helium's simplified automation commands to this newly created Selenium session.
    return helium.start_chrome(headless=False, options=chrome_options)


def initialize_agent(model):
    """Initialize the CodeAgent with the specified model."""
    # CodeAgent is an LLM agent that thinks in Python code: it writes Python code blocks to call tools and manipulate data.
    return CodeAgent(
        # The list of tools made available to the agent.
        # - WebSearchTool(): built-in web search.
        # - go_back, close_popups, search_item_ctrl_f: our custom @tool decorated functions.
        tools=[WebSearchTool(), go_back, close_popups, search_item_ctrl_f],
        
        # The language model instance used to reason and generate code.
        model=model,
        
        # By default, smolagents sandboxes code execution and blocks unauthorized library imports.
        # This parameter explicitly whitelists the 'helium' library so the agent's code can call Helium commands.
        additional_authorized_imports=["helium"],
        
        # Functions to invoke automatically at the end of each step (here, our screenshot capturing callback).
        step_callbacks=[save_screenshot],
        
        # Safety limit: prevents the agent from running in an infinite loop; aborts if not completed in 20 steps.
        max_steps=20,
        
        # Verbosity level controls console logging: 2 provides detailed execution traces, thoughts, and tool inputs/outputs.
        verbosity_level=2,
    )


# ==============================================================================
# 7. AGENT SYSTEM INSTRUCTIONS
# ==============================================================================

# Detailed guidelines injected into the agent prompt explaining how to interact with Helium,
# how to navigate, how to handle pop-ups, when to scroll, and how to conclude the task.
helium_instructions = """
Use your web_search tool when you want to get Google search results.
Then you can use helium to access websites. Don't use helium for Google search, only for navigating websites!
Don't bother about the helium driver, it's already managed.
We've already ran "from helium import *"
Then you can go to pages!
Code:
```py
go_to('github.com/trending')
```<end_code>
You can directly click clickable elements by inputting the text that appears on them.
Code:
```py
click("Top products")
```<end_code>
If it's a link:
Code:
```py
click(Link("Top products"))
```<end_code>
If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your screenshot.
Never try to login in a page.
To scroll up or down, use scroll_down or scroll_up with as an argument the number of pixels to scroll from.
Code:
```py
scroll_down(num_pixels=1200) # This will scroll one viewport down
```<end_code>
When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
Code:
```py
close_popups()
```<end_code>
You can use .exists() to check for the existence of an element. For example:
Code:
```py
if Text('Accept cookies?').exists():
    click('I accept')
```<end_code>
Proceed in several steps rather than trying to solve the task in one shot.
And at the end, only when you have your answer, return your final answer.
Code:
```py
final_answer("YOUR_ANSWER_HERE")
```<end_code>
If pages seem stuck on loading, you might have to wait, for instance `import time` and run `time.sleep(5.0)`. But don't overuse this!
To list elements on page, DO NOT try code-based element searches like 'contributors = find_all(S("ol > li"))': just look at the latest screenshot you have and read it visually, or use your tool search_item_ctrl_f.
Of course, you can act on buttons like a user would do when navigating.
After each code blob you write, you will be automatically provided with an updated screenshot of the browser and the current browser url.
But beware that the screenshot will only be taken at the end of the whole action, it won't see intermediate states.
Don't kill the browser.
When you have modals or cookie banners on screen, you should get rid of them before you can click anything else.
"""


# ==============================================================================
# 8. MAIN EXECUTION ENTRY POINT
# ==============================================================================

def main():
    # Load environment variables from a local '.env' file.
    # For example to use an OpenAI model, create a local .env file with OPENAI_API_KEY="<your_open_ai_key_here>"
    load_dotenv() 

    # Parse command line arguments provided by the user (or fall back to defaults).
    args = parse_arguments()

    # Initialize the LLM instance using smolagents' loader, passing the model type and model ID.
    model = load_model(args.model_type, args.model_id)

    # Declare 'driver' as a global variable so custom tool functions (e.g. search_item_ctrl_f, go_back, close_popups)
    # defined in the outer scope can directly access and control this browser instance.
    global driver
    
    # Launch Chrome and get the driver reference.
    driver = initialize_driver()
    
    # Create the CodeAgent configured with the model and tools.
    agent = initialize_agent(model)

    # agent.python_executor evaluates Python code inside the agent's isolated execution scope.
    # Pre-running "from helium import *" imports Helium functions (click, write, go_to, etc.) into the agent's
    # runtime namespace so the agent can use them directly in generated code without writing import statements.
    agent.python_executor("from helium import *")
    
    # Start the agent execution loop with the combined task prompt and Helium operational instructions.
    agent.run(args.prompt + helium_instructions)


# Standard Python idiom: checks if the script is being run directly by Python (as opposed to imported as a module).
# If run directly (__name__ == "__main__"), call the main() function to start execution.
if __name__ == "__main__":
    main()
