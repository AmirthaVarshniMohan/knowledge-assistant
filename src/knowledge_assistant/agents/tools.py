"""Custom tools for AI Agent workflows: Safe Calculator, Knowledge Retriever, and System Inspector."""

import ast
import operator
from typing import Any, Callable, Dict, List, Optional
from loguru import logger
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from knowledge_assistant.core.config import get_settings
from knowledge_assistant.rag.retriever import Retriever
from knowledge_assistant.rag.vector_store import BaseVectorStore


class SafeCalculator:
    """Safe mathematical expression evaluator using Python AST (Abstract Syntax Tree).
    
    Prevents security vulnerabilities associated with dangerous eval().
    """

    OPERATORS: Dict[type, Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @classmethod
    def evaluate(cls, expression: str) -> float:
        """Parse and evaluate a math expression safely."""
        clean_expr = expression.strip().replace(",", "")
        try:
            tree = ast.parse(clean_expr, mode="eval")
            return float(cls._eval_node(tree.body))
        except Exception as e:
            raise ValueError(f"Invalid or unsafe mathematical expression '{expression}': {e}")

    @classmethod
    def _eval_node(cls, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in cls.OPERATORS:
                raise ValueError(f"Unsupported binary operator: {op_type}")
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            return cls.OPERATORS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in cls.OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type}")
            operand = cls._eval_node(node.operand)
            return cls.OPERATORS[op_type](operand)
        else:
            raise ValueError(f"Unsupported syntax expression node: {type(node)}")


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="A clean mathematical expression string to calculate (e.g., '15 * 500' or '(1000 - 250) / 4')"
    )


@tool("calculator", args_schema=CalculatorInput)
def calculate(expression: str) -> str:
    """Perform precise mathematical arithmetic calculations safely."""
    try:
        result = SafeCalculator.evaluate(expression)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


def create_knowledge_tool(retriever: Retriever) -> BaseTool:
    """Create a LangChain tool for querying the internal Enterprise Knowledge Base."""

    class KnowledgeQueryInput(BaseModel):
        query: str = Field(description="Search term or question to look up in internal enterprise documents")

    @tool("search_enterprise_knowledge", args_schema=KnowledgeQueryInput)
    def search_knowledge(query: str) -> str:
        """Search company policies, security guidelines, and internal documentation."""
        logger.debug(f"[Agent Tool] Querying knowledge base for: '{query}'")
        results = retriever.retrieve(query=query, k=3)
        if not results:
            return "No matching internal documents found."

        snippets = []
        for r in results:
            file_name = r.metadata.get("file_name", "unknown")
            page = r.metadata.get("page", 1)
            snippets.append(f"[{file_name} - Page {page}]: {r.content.strip()}")

        return "\n\n".join(snippets)

    return search_knowledge


def create_system_status_tool(vector_store: Optional[BaseVectorStore] = None) -> BaseTool:
    """Create a tool that provides real-time system health and document metrics."""

    @tool("get_system_status")
    def get_system_status() -> str:
        """Inspect the current health, indexed document count, and environment status of the knowledge assistant."""
        settings = get_settings()
        count = vector_store.count() if vector_store else 0
        return (
            f"Application: {settings.app_name}\n"
            f"Environment: {settings.environment}\n"
            f"Active Vector Collection: {settings.collection_name}\n"
            f"Total Indexed Chunks: {count}"
        )

    return get_system_status


def get_default_tools(
    retriever: Retriever,
    vector_store: Optional[BaseVectorStore] = None,
) -> List[BaseTool]:
    """Return the standard toolkit for the Enterprise AI Agent."""
    return [
        calculate,
        create_knowledge_tool(retriever),
        create_system_status_tool(vector_store),
    ]
