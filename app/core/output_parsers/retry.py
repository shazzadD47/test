from typing import TypeVar

from langchain_classic.output_parsers import (
    RetryOutputParser as LangchainRetryOutputParser,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue

T = TypeVar("T")


class RetryOutputParser(LangchainRetryOutputParser):
    def parse_with_prompt(
        self, completion: str | AIMessage, prompt_value: PromptValue
    ) -> T:
        """Parse the output of an LLM call using a wrapped parser.

        Args:
            completion: The chain completion to parse.
            prompt_value: The prompt to use to parse the completion.

        Returns:
            The parsed completion.
        """
        retries = 0

        while retries <= self.max_retries:
            try:
                if isinstance(completion, AIMessage):
                    completion = completion.content

                return self.parser.parse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise e
                else:
                    retries += 1
                    if self.legacy and hasattr(self.retry_chain, "run"):
                        completion = self.retry_chain.run(
                            prompt=prompt_value.to_string(),
                            completion=completion,
                            error=repr(e),
                        )
                    else:
                        completion = self.retry_chain.invoke(
                            {
                                "prompt": prompt_value.to_string(),
                                "completion": completion,
                            }
                        )

        raise OutputParserException("Failed to parse")

    async def aparse_with_prompt(
        self, completion: str | AIMessage, prompt_value: PromptValue
    ) -> T:
        """Parse the output of an LLM call using a wrapped parser.

        Args:
            completion: The chain completion to parse.
            prompt_value: The prompt to use to parse the completion.

        Returns:
            The parsed completion.
        """
        retries = 0

        while retries <= self.max_retries:
            try:
                if isinstance(completion, AIMessage):
                    completion = completion.content

                return await self.parser.aparse(completion)
            except OutputParserException as e:
                if retries == self.max_retries:
                    raise e
                else:
                    retries += 1
                    if self.legacy and hasattr(self.retry_chain, "arun"):
                        completion = await self.retry_chain.arun(
                            prompt=prompt_value.to_string(),
                            completion=completion,
                            error=repr(e),
                        )
                    else:
                        completion = await self.retry_chain.ainvoke(
                            {
                                "prompt": prompt_value.to_string(),
                                "completion": completion,
                            }
                        )

        raise OutputParserException("Failed to parse")
