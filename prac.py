import inspect
import json
import os

from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Dict, List, Tuple


# 加载 .env 中的环境变量
load_dotenv(Path(__file__).resolve().parent / ".env")

# 初始化模型 API 客户端
openai_client = OpenAI(api_key=os.environ["GROQ_API_KEY"],base_url="https://api.groq.com/openai/v1")


# TODO：告诉模型它的角色、有哪些工具，以及工具调用格式

SYSTEM_PROMPT = """
You are a coding assistant helping with coding tasks.

Available tools:
{tool_list_repr}

An external Python program executes tools based on your text replies.
Write tool requests as plain text, not native API tool calls.

When you need a tool, reply with exactly one line:
tool: TOOL_NAME({{JSON_ARGS}})

For example:
tool: read_file({{"filename":"t1.py"}})

Do not include explanations, surrounding quotes, or Markdown fences
in a tool request.

Never guess or invent file contents. If you need to know what is in a file,
call read_file first. Only call one tool per reply, then wait for tool_result.

After receiving tool_result(...), use the result to continue.
If no tool is needed, respond normally.
"""

# 终端输出颜色
YOU_COLOR = "\u001b[94m"
ASSISTANT_COLOR = "\u001b[93m"
RESET_COLOR = "\u001b[0m"


# 已提供的辅助函数：将文件路径转换为绝对路径
def resolve_abs_path(path_str: str) -> Path:
    """
    例如：
    file.py -> /Users/home/project/file.py
    """
    path = Path(path_str).expanduser()

    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    return path


# --------------------------------------------------
# 三个文件工具
# --------------------------------------------------

def read_file_tool(filename: str) -> Dict[str, Any]:
    """
    Gets the full content of a file provided by the user.
    :param filename: The name of the file to read.
    :return: The full content of the file.
    """
    full_path = resolve_abs_path(filename)
    print(full_path)
    with open(str(full_path), "r") as f:
        content = f.read()
    return {
        "file_path": str(full_path),
        "content": content
    }
    


def list_files_tool(path: str) -> Dict[str, Any]:
    """
    Lists the files in a directory provided by the user.
    :param path: The path to a directory to list files from.
    :return: A list of files in the directory.
    """
    full_path = resolve_abs_path(path)
    all_files = []
    for item in full_path.iterdir():
        all_files.append({
            "filename": item.name,
            "type": "file" if item.is_file() else "dir"
        })
    return {
        "path": str(full_path),
        "files": all_files
    }


def edit_file_tool(
    path: str,
    old_str: str,
    new_str: str
) -> Dict[str, Any]:
    """
    Replaces first occurrence of old_str with new_str in file.
    If old_str is empty, create/overwrite file with new_str.
    :param path: The path to the file to edit.
    :param old_str: The string to replace.
    :param new_str: The string to replace with.
    :return: A dictionary with the path and action taken.
    """
    full_path = resolve_abs_path(path)
    if old_str == "":
        full_path.write_text(new_str, encoding="utf-8")
        return {
            "path": str(full_path),
            "action": "created_file"
        }
    original = full_path.read_text(encoding="utf-8")
    if original.find(old_str) == -1:
        return {
            "path": str(full_path),
            "action": "old_str not found"
        }
    edited = original.replace(old_str, new_str, 1)
    full_path.write_text(edited, encoding="utf-8")
    return{
        "path": str(full_path),
        "action": "edited"
    }


# 工具注册表：工具名称 -> 对应的 Python 函数
TOOL_REGISTRY = {
    "read_file": read_file_tool,
    "list_files": list_files_tool,
    "edit_file": edit_file_tool
}


# --------------------------------------------------
# 工具说明与 system prompt
# --------------------------------------------------

def get_tool_str_representation(tool_name: str) -> str:
    # TODO：生成一个工具的文字说明
    # 包括：名称、docstring、参数签名
    tool = TOOL_REGISTRY[tool_name]
    return f"""
    Name: {tool_name}
    Description: {tool.__doc__}
    Signature: {inspect.signature(tool)}
    """



def get_full_system_prompt():
    # TODO：将所有工具的说明填入 SYSTEM_PROMPT
    tool_str_repr = ""
    for tool_name in TOOL_REGISTRY:
        tool_str_repr += "TOOL\n===" + get_tool_str_representation(tool_name)
        tool_str_repr += f"\n{'='*15}\n"
    return SYSTEM_PROMPT.format(tool_list_repr = tool_str_repr)


# --------------------------------------------------
# 解析模型输出与调用模型
# --------------------------------------------------

def extract_tool_invocations(
    text: str
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Return list of (tool_name, args) requested in
    'tool: name({...})' lines.

    The parser expects single-line, compact JSON
    in parentheses.
    """
    invocations = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("tool:"):
            continue
        try:
            after = line[len("tool:"):].strip()
            # after = 'read_file({"filename":"t1.py"})'
            name, rest = after.split("(", 1)
            name = name.strip()
            if not rest.endswith(")"):
                continue
            json_str = rest[:-1].strip()
            args = json.loads(json_str)
            invocations.append((name, args))
        except Exception:
            continue
    return invocations
            
            
        


# def execute_llm_call(conversation: List[Dict[str, str]]):
#     response = openai_client.chat.completions.create(
#         model = "openai/gpt-oss-120b",
#         messages = conversation,
#         max_completion_tokens = 2000
#     )
#     return response.choices[0].message.content
def execute_llm_call(conversation: List[Dict[str, str]]):
    response = openai_client.chat.completions.create(
        model = "qwen/qwen3.8-27b",
        messages = conversation,
        max_completion_tokens = 800
    )
    choice = response.choices[0]
    if not choice.message.content:
        print("[debug] finish_reason:", choice.finish_reason)
        print("[debug] tool_calls:", choice.message.tool_calls)
        print("[debug] reasoning:", getattr(choice.message, "reasoning", None))
    return choice.message.content or ""


# --------------------------------------------------
# Agent 核心循环——老师接下来先实现这里
# --------------------------------------------------

def run_coding_agent_loop():
    # TODO：
    # 1. 初始化 conversation，放入 system prompt
    # 2. 接收用户输入
    # 3. 调用模型
    # 4. 检查模型是否请求使用工具
    # 5. 如果需要：执行工具、保存结果、再次调用模型
    # 6. 如果不需要：展示回复，等待下一个用户输入
    print(get_full_system_prompt());
    conversation = [{
        "role": "system",
        "content": get_full_system_prompt()
    }]
    while True:
        try:
            user_input = input(f"{YOU_COLOR}You{RESET_COLOR}: ")
        except (KeyboardInterrupt, EOFError):
            break
        conversation.append({
            "role":"user",
            "content": user_input.strip()
        })
        
        while True:
            try:
                assistant_response = execute_llm_call(conversation)
            except Exception as e:
                print(f"[error] LLM call failed: {type(e).__name__}: {e}")
                break
            tool_invocations = extract_tool_invocations(assistant_response)
            if not tool_invocations:
                print(f"{ASSISTANT_COLOR}Assistant{RESET_COLOR}: {assistant_response}")
                conversation.append(
                    {"role": "assistant",
                    "content": assistant_response}
                )
                break
            print(f"{ASSISTANT_COLOR}Assistant{RESET_COLOR}: {assistant_response}")
            conversation.append({
                "role": "assistant",
                "content": assistant_response
            })
            for name, args in tool_invocations:
                print(name, args)
                tool = TOOL_REGISTRY[name]
                resp = []
                try:
                    if name == "read_file":
                        resp = tool(args.get("filename", "."))
                    elif name == "list_files":
                        resp = tool(args.get("path", "."))
                    elif name == "edit_file":
                        resp = tool(args.get("path", "."), 
                                    args.get("old_str", ""), 
                                    args.get("new_str", ""))
                except Exception as e:
                    resp = {"error": f"{type(e).__name__}: {e}"}
            conversation.append({
                "role": "user",
                "content": f"tool_result({json.dumps(resp)})"
            })

if __name__ == "__main__":
    run_coding_agent_loop()