"""
AI Code Bug Fixer - LangChain Agent with Ollama
LangChain 1.0.3 compatible version
"""

import ast
import re
from typing import List, Dict
from pydantic import BaseModel, Field

from langchain_community.llms import Ollama



# -----------------------------
# Code Analyzer
# -----------------------------

class CodeAnalyzer:
    """Analyzes code for common bugs and issues"""

    @staticmethod
    def check_python_syntax(code: str) -> Dict:
        try:
            ast.parse(code)
            return {"valid": True, "errors": []}
        except SyntaxError as e:
            return {
                "valid": False,
                "errors": [{
                    "type": "SyntaxError",
                    "line": e.lineno,
                    "message": e.msg,
                    "text": e.text
                }]
            }

    @staticmethod
    def check_imports(code: str) -> Dict:
        issues = []
        lines = code.split("\n")

        imported_modules = set()
        for line in lines:
            if line.strip().startswith(("import ", "from ")):
                match = re.search(r"import\s+(\w+)", line)
                if match:
                    imported_modules.add(match.group(1))

        code_without_imports = "\n".join(
            l for l in lines
            if not l.strip().startswith(("import ", "from "))
        )

        for module in imported_modules:
            if module not in code_without_imports:
                issues.append(f"Potentially unused import: {module}")

        return {"issues": issues}

    @staticmethod
    def check_common_bugs(code: str) -> List[str]:
        bugs = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            if "==" in line and not any(k in line for k in ("if", "while", "assert")):
                bugs.append(f"Line {i}: Possible unintended comparison (==)")

            if line.strip().startswith("except") and i < len(lines):
                if lines[i].strip() == "pass":
                    bugs.append(f"Line {i}: Empty except block")

            if "def " in line and ("=[]" in line or "={}" in line):
                bugs.append(f"Line {i}: Mutable default argument")

        return bugs


# -----------------------------
# Tool Input Schemas
# -----------------------------

class FilePathInput(BaseModel):
    file_path: str = Field(..., description="Path to the Python file")


# -----------------------------
# Bug Fixer Agent
# -----------------------------

class CodeBugFixer:

    def __init__(self, model_name: str = "llama3.2"):
        self.llm = Ollama(model=model_name, temperature=0.1)
        self.analyzer = CodeAnalyzer()
        self.agent = self._create_agent()

    # ---------- TOOLS ----------

    def read_code(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            return f.read()

    def analyze_code(self, file_path: str) -> str:
        code = self.read_code(file_path)

        syntax = self.analyzer.check_python_syntax(code)
        imports = self.analyzer.check_imports(code)
        bugs = self.analyzer.check_common_bugs(code)

        report = [f"Analysis Report for {file_path}\n"]

        if not syntax["valid"]:
            report.append("SYNTAX ERRORS:")
            for e in syntax["errors"]:
                report.append(f"Line {e['line']}: {e['message']}")
        else:
            report.append("✓ No syntax errors")

        if imports["issues"]:
            report.append("\nIMPORT ISSUES:")
            report.extend(imports["issues"])

        if bugs:
            report.append("\nCOMMON BUGS:")
            report.extend(bugs)

        if syntax["valid"] and not imports["issues"] and not bugs:
            report.append("\n✓ No obvious issues detected")

        return "\n".join(report)

    def fix_code(self, file_path: str) -> str:
        original = self.read_code(file_path)
        analysis = self.analyze_code(file_path)

        if "No obvious issues detected" in analysis:
            return "No fixes needed."

        prompt = f"""
Fix the following Python code.

ISSUES:
{analysis}

CODE:
```python
{original}
            fixed = self.llm.invoke(prompt).strip()
            fixed = re.sub(r"```python|```", "", fixed).strip()

            output_path = file_path.replace(".py", "_fixed.py")
            with open(output_path, "w") as f:
                f.write(fixed)

            return f"Fixed code saved to {output_path}"

        # ---------- AGENT ----------

        def _create_agent(self) -> AgentExecutor:
            tools = [
                StructuredTool.from_function(
                    name="ReadCode",
                    description="Read a Python file",
                    func=self.read_code,
                    args_schema=FilePathInput,
                ),
                StructuredTool.from_function(
                    name="AnalyzeCode",
                    description="Analyze Python code for bugs",
                    func=self.analyze_code,
                    args_schema=FilePathInput,
                ),
                StructuredTool.from_function(
                    name="FixCode",
                    description="Fix bugs in Python code",
                    func=self.fix_code,
                    args_schema=FilePathInput,
                ),
            ]

            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert Python debugging agent."),
                ("human", "{input}"),
                ("ai", "{agent_scratchpad}")
            ])

            agent = create_react_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt,
            )

            return AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=5,
            )

        # ---------- RUN ----------

        def process_file(self, file_path: str, auto_fix: bool = False):
            query = (
                f"Analyze and fix the file at {file_path}"
                if auto_fix
                else f"Analyze the file at {file_path}"
            )
            return self.agent.invoke({"input": query})
