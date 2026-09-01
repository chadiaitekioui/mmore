"""
Example implementation:
RAG pipeline.
Integrates Milvus retrieval with HuggingFace text generation.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, cast

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)

from ..utils import load_config
from .judge import JUDGE_OUTPUT_KEYS, JudgeConfig, LLMJudge, retrieve_with_judge
from .judge.llm import judge_llm_from_config
from .llm import LLM, LLMConfig
from .model.vision import (
    BaseMultimodalLLM,
    aggregate_image_paths,
    get_multimodal_llm,
    load_images_from_paths,
)
from .retriever import Retriever, RetrieverConfig
from .types import MMOREInput, MMOREOutput

DEFAULT_PROMPT = """\
Use the following context to answer the questions. If none of the context answer the question, just say you don't know.

Context:
{context}
"""

PRIVACY_OUTPUT_KEYS = ("privacy_report", "sanitized_context")
PRIVACY_CHUNKS_KEY = "sanitized_chunks"

# Answers the privacy gate's interrupt payloads
PrivacyApprover = Callable[[dict], object]


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""

    retriever: RetrieverConfig
    llm: LLMConfig = field(default_factory=lambda: LLMConfig(llm_name="gpt2"))
    system_prompt: str = DEFAULT_PROMPT
    max_images_per_request: int = 20
    judge: Optional[JudgeConfig] = None


class RAGPipeline:
    """Main RAG pipeline combining retrieval and generation."""

    retriever: Retriever
    llm: Optional[BaseChatModel]
    prompt_template: Union[str, ChatPromptTemplate]

    def __init__(
        self,
        retriever: Retriever,
        prompt_template: Union[str, ChatPromptTemplate],
        llm: Optional[BaseChatModel] = None,
        use_vision: bool = False,
        multimodal_llm: Optional[BaseMultimodalLLM] = None,
        max_images_per_request: int = 20,
        judge: Optional[LLMJudge] = None,
        privacy_graph: Optional[Any] = None,
        privacy_approver: Optional[PrivacyApprover] = None,
    ):
        if privacy_graph is None and use_vision and multimodal_llm is None:
            raise ValueError("Vision mode requires a multimodal LLM.")
        # Get modules
        self.retriever = retriever
        self.prompt = prompt_template
        self.llm = llm
        self.use_vision = use_vision
        self.multimodal_llm = multimodal_llm
        self.max_images_per_request = max_images_per_request
        self.judge = judge
        self.privacy_graph = privacy_graph
        # Answers the gate's interrupts when the privacy config is interactive
        self.privacy_approver = privacy_approver

        # Build the rag chain
        self.rag_chain = RAGPipeline._build_chain(
            self.retriever,
            RAGPipeline.format_docs,
            self.prompt,
            self.llm,
            use_vision=self.use_vision,
            multimodal_llm=self.multimodal_llm,
            max_images_per_request=self.max_images_per_request,
            judge=self.judge,
            privacy_graph=self.privacy_graph,
            privacy_approver=self.privacy_approver,
        )

    def __str__(self):
        return str(self.rag_chain)

    @classmethod
    def from_config(
        cls,
        config: str | RAGConfig,
        privacy_graph: Optional[Any] = None,
        privacy_approver: Optional[PrivacyApprover] = None,
    ):
        if isinstance(config, str):
            config = load_config(config, RAGConfig)

        retriever = Retriever.from_config(config.retriever)
        if privacy_graph is not None:
            if config.llm.use_vision:
                raise ValueError("Privacy mode and vision mode are mutually exclusive.")
            llm: Optional[BaseChatModel] = None
            multimodal_llm = None
        elif config.llm.use_vision:
            llm = None
            multimodal_llm = get_multimodal_llm(config.llm)
            multimodal_llm._load()
        else:
            llm = LLM.from_config(config.llm)
            multimodal_llm = None
        judge = (
            LLMJudge(llm=judge_llm_from_config(config.judge.llm), config=config.judge)
            if config.judge
            else None
        )
        chat_template = ChatPromptTemplate.from_messages(
            [("system", config.system_prompt), ("human", "{input}")]
        )

        return cls(
            retriever,
            chat_template,
            llm,
            use_vision=config.llm.use_vision,
            multimodal_llm=multimodal_llm,
            max_images_per_request=config.max_images_per_request,
            judge=judge,
            privacy_graph=privacy_graph,
            privacy_approver=privacy_approver,
        )

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Format documents for prompt."""
        return "\n\n".join(
            f"[{doc.metadata['rank']}] {doc.page_content}" for doc in docs
        )

    @staticmethod
    def _build_chain(
        retriever,
        format_docs,
        prompt,
        llm,
        use_vision=False,
        multimodal_llm=None,
        max_images_per_request=20,
        judge=None,
        privacy_graph=None,
        privacy_approver=None,
    ) -> Runnable:
        validate_input = RunnableLambda(
            lambda x: MMOREInput.model_validate(x).model_dump()
        )

        def make_output(x):
            """Validate the output of the LLM and keep only the actual answer of the assistant"""
            res_dict = MMOREOutput.model_validate(x).model_dump()
            if use_vision and multimodal_llm is not None:
                res_dict["image_paths"] = aggregate_image_paths(x["docs"])[
                    :max_images_per_request
                ]
            res_dict["answer"] = res_dict["answer"].split("<|im_start|>assistant\n")[-1]
            # Expose formatted context and judge correction logs in the API response (context is not on MMOREOutput).
            for key in (
                "context",
                *JUDGE_OUTPUT_KEYS,
                *PRIVACY_OUTPUT_KEYS,
                PRIVACY_CHUNKS_KEY,
            ):
                if key in x:
                    res_dict[key] = x[key]

            return res_dict

        validate_output = RunnableLambda(make_output)

        def answer_with_vision(x: Dict[str, Any]) -> str:
            images = load_images_from_paths(
                aggregate_image_paths(x["docs"]), max_images=max_images_per_request
            )
            # Keep the chat roles instead of flattening the prompt into one blob.
            system_parts: List[str] = []
            user_parts: List[str] = []
            for message in prompt.invoke(
                {"context": x["context"], "input": x["input"]}
            ).to_messages():
                is_system = getattr(message, "type", None) == "system"
                (system_parts if is_system else user_parts).append(str(message.content))
            return multimodal_llm.invoke_with_images(
                text="\n\n".join(part for part in user_parts if part),
                images=images,
                system_prompt="\n\n".join(part for part in system_parts if part)
                or None,
            )

        # Only retrieval differs (retriever vs judge); format context and generate answer unchanged.
        if judge is not None:
            # retrieve with judge
            def retrieval_with_judge(state: Dict[str, Any]) -> Dict[str, Any]:
                return retrieve_with_judge(retriever, judge, state)

            retrieval_step: Runnable = RunnableLambda(retrieval_with_judge)
        else:
            # retrieve without judge
            retrieval_step = RunnablePassthrough.assign(docs=retriever)

        with_context = retrieval_step.assign(context=lambda x: format_docs(x["docs"]))
        if privacy_graph is not None:
            # Privacy mode swaps only the answer step: the verified answer comes
            # from the privacy graph driven over the retrieved chunks.
            answer_step: Runnable = RunnableLambda(
                RAGPipeline._privacy_answer_step(privacy_graph, privacy_approver)
            )
            core_chain = with_context | answer_step
        elif use_vision and multimodal_llm is not None:
            core_chain = with_context.assign(answer=RunnableLambda(answer_with_vision))
        else:
            if llm is None:
                raise ValueError("RAGPipeline needs an LLM when privacy mode is off.")
            core_chain = with_context.assign(answer=prompt | llm | StrOutputParser())

        return validate_input | core_chain | validate_output

    @staticmethod
    def _privacy_answer_step(privacy_graph, privacy_approver=None):
        """Answer step that routes the retrieved chunks through the privacy graph.

        Returns the chain state updated with the verified ``answer`` and the
        PII-free report fields, mirroring the ``.assign(answer=...)`` it replaces.
        """
        from dataclasses import asdict

        from ..privacy.runner import run_privacy_query

        def step(x: Dict[str, Any]) -> Dict[str, Any]:
            docs: List[Document] = x["docs"]
            raw_chunks = [doc.page_content for doc in docs]
            result = run_privacy_query(
                privacy_graph, x["input"], raw_chunks, approver=privacy_approver
            )
            updated = dict(x)
            updated["answer"] = result.answer
            updated[PRIVACY_CHUNKS_KEY] = list(result.sanitized_chunks)
            updated["sanitized_context"] = "\n\n".join(
                chunk for chunk in result.sanitized_chunks if chunk
            ).strip()
            if result.record is not None:
                updated["privacy_report"] = asdict(result.record)
            return updated

        return step

    def __call__(
        self,
        queries: Dict[str, Any] | List[Dict[str, Any]],
        return_dict: bool = False,
        config: Optional[RunnableConfig] = None,
    ) -> List[Dict[str, Any]]:
        if isinstance(queries, dict):
            queries_list = [queries]
        else:
            queries_list = queries

        if self.use_vision and self.multimodal_llm is not None:
            # Vision generation is memory-heavy: keep the batch sequential.
            config = cast(RunnableConfig, {"max_concurrency": 1, **(config or {})})
        results = self.rag_chain.batch(queries_list, config=config)

        if return_dict:
            return results
        else:
            return [result["answer"] for result in results]
