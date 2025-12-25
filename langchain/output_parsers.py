from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)

class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int | None = Field(description="The rating of the product", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
    key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")

system_prompt = (
        "You are a helpful and knowledgeable assistant. "
        "Answer questions and engage in conversation based on your general knowledge if you do not find linked tool"
        "You do not have access to external tools for information retrieval."
    )

#LangChain automatically uses ProviderStrategy when you pass a schema type directly to create_agent.response_format
#For models that don’t support native structured output, LangChain uses tool calling to achieve the same result. This works with all models that support tool calling, which is most modern models.

agent = create_agent(
    model=llm,
    tools=[],
    response_format=ToolStrategy(ProductReview),
    debug=False,
    system_prompt=system_prompt
)
result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})
print(result["structured_response"])