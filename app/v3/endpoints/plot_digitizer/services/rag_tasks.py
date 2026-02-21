from uuid import uuid4

from anthropic import RateLimitError as AnthropicRateLimitError
from langfuse import observe
from openai import RateLimitError as OpenAIRateLimitError

from app.configs import settings
from app.redis import redis_client
from app.utils.llms import batch_invoke_chain_with_retry
from app.utils.tracing import setup_langfuse_handler
from app.v3.endpoints.plot_digitizer.configs import settings as pd_settings
from app.v3.endpoints.plot_digitizer.constants import (
    MAX_RETRIES,
    rephrase_chain,
)
from app.v3.endpoints.plot_digitizer.exceptions import (
    QuestionRephrasingFailed,
)
from app.v3.endpoints.plot_digitizer.logging import celery_logger as logger

REDIS_LIVE_TIME = settings.CACHE_DAY * 24 * 60 * 60


@observe()
def rephrase_question(
    questions: list[str],
    keys: list[str] = None,
    langfuse_session_id: str = None,
) -> list[str] | list[dict[str, str]]:
    if langfuse_session_id is None:
        langfuse_session_id = uuid4().hex
    langfuse_handler = setup_langfuse_handler(langfuse_session_id)

    batch_size = pd_settings.QUERY_REPHRASE_BATCH_SIZE
    if keys:
        rephrased_questions = dict(zip(keys, questions))
    else:
        rephrased_questions = dict(enumerate(questions))

    indices_to_rephrase = []
    for key, question in rephrased_questions.items():
        question_key = f"paraphrased_{question}"
        if question_key in redis_client:
            rephrased_question = redis_client.getex(question_key, ex=REDIS_LIVE_TIME)
            rephrased_questions[key] = rephrased_question
        else:
            indices_to_rephrase.append(key)

    selected_questions = [rephrased_questions[key] for key in indices_to_rephrase]

    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            rephrased_questions_for_selected_questions = batch_invoke_chain_with_retry(
                chain=rephrase_chain,
                input=[{"question": question} for question in selected_questions],
                config={"max_concurrency": batch_size, "callbacks": [langfuse_handler]},
                max_retries=1,
            )
            break
        except AnthropicRateLimitError:
            logger.exception("Anthropic rate limit exceeded.")
            batch_size = max(batch_size // 2, 1)
            retry_count += 1
            if retry_count == MAX_RETRIES:
                raise QuestionRephrasingFailed()
            continue
        except OpenAIRateLimitError:
            logger.exception("OpenAI rate limit exceeded.")
            batch_size = max(batch_size // 2, 1)
            retry_count += 1
            if retry_count == MAX_RETRIES:
                raise QuestionRephrasingFailed()
            continue
        except Exception:
            logger.exception("An error occurred while rephrasing the question.")
            retry_count += 1
            if retry_count == MAX_RETRIES:
                raise QuestionRephrasingFailed()
            continue
    for i, key in enumerate(indices_to_rephrase):
        rephrased_questions[key] = rephrased_questions_for_selected_questions[i]

    if keys is None:
        only_rephrased_questions = {
            rephrased_questions[i] for i in range(len(questions))
        }
        return list(only_rephrased_questions)
    else:
        return [{key: rephrased_questions[key]} for key in keys]
