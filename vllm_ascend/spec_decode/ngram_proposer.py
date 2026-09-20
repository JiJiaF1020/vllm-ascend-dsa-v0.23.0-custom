import torch
from vllm.logger import init_logger
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

from vllm_ascend.dsa_offload.synthetic_draft import synthetic_draft_enabled
from vllm_ascend.utils import vllm_version_is

logger = init_logger(__name__)


class AscendNgramProposer(NgramProposer):
    def __init__(self, vllm_config, runner):
        self.runner = runner
        self.synthetic_draft = synthetic_draft_enabled(vllm_config)
        if self.synthetic_draft and not vllm_version_is("0.23.0"):
            raise ValueError("[TO BE REMOVE] synthetic draft is only wired for vLLM 0.23.0")
        if self.synthetic_draft:
            # 不分配ngram搜索scratch，不触发其Numba预热（长上下文会分配大矩阵）。
            self.k = vllm_config.speculative_config.num_speculative_tokens
            self.max_model_len = vllm_config.model_config.max_model_len
            logger.warning("DSA_SYNTHETIC_DRAFT_ENABLED drafts=3 physical_mtp_layers=0 rejection=standard NOT_MTP_PERFORMANCE")
        else:
            super().__init__(vllm_config)

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens,
        with_prefill=None,
        in_graph_capturing=None,
        num_reqs=None,
        num_tokens_across_dp=None,
        aclgraph_runtime_mode=None,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
    ):
        pass

    if vllm_version_is("0.23.0"):

        def propose(
            self,
            sampled_token_ids: list[list[int]],
            num_tokens_no_spec=None,
            token_ids_cpu=None,
            slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
        ) -> list[list[int]]:
            input_batch = self.runner.input_batch
            valid_ngram_requests = []
            for i, sampled_ids in enumerate(sampled_token_ids):
                num_sampled_ids = len(sampled_ids)
                if not num_sampled_ids:
                    continue

                req_id = input_batch.req_ids[i]
                if req_id in input_batch.spec_decode_unsupported_reqs:
                    continue

                num_tokens = input_batch.num_tokens_no_spec[i]
                if num_tokens >= input_batch.max_model_len:
                    # Skip requests that have already reached the max model length.
                    continue

                start_idx = input_batch.num_tokens_no_spec[i]
                end_idx = start_idx + num_sampled_ids
                input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids

                valid_ngram_requests.append(i)

            if self.synthetic_draft:
                # [TO BE REMOVE] 保留上面的有效请求/历史输入维护，不做ngram搜索。
                # sampled_token_ids来自正常target采样；不假造accepted长度或request身份。
                drafts = [[] for _ in sampled_token_ids]
                for i in valid_ngram_requests:
                    remaining = input_batch.max_model_len - (
                        int(input_batch.num_tokens_no_spec[i]) + len(sampled_token_ids[i]))
                    count = min(self.k, max(0, remaining))
                    drafts[i] = [int(sampled_token_ids[i][-1])] * count
                logger.info("DSA_SYNTHETIC_DRAFT_PROPOSE requests=%s draft_lengths=%s rejection=standard",
                            len(drafts), [len(row) for row in drafts])
                return drafts

            return self.batch_propose(
                len(sampled_token_ids),
                valid_ngram_requests,
                input_batch.num_tokens_no_spec,
                input_batch.token_ids_cpu,
            )

    else:

        def propose(  # type: ignore[misc]
            self,
            num_speculative_tokens: int,
            sampled_token_ids: list[list[int]],
            num_tokens_no_spec=None,
            token_ids_cpu=None,
            slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
        ) -> list[list[int]]:
            assert num_speculative_tokens <= self.k
            assert num_tokens_no_spec is not None
            assert token_ids_cpu is not None

            valid_ngram_requests = []
            for i, sampled_ids in enumerate(sampled_token_ids):
                num_sampled_ids = len(sampled_ids)
                if not num_sampled_ids:
                    continue

                num_tokens = num_tokens_no_spec[i]
                if num_tokens >= self.max_model_len:
                    # Skip requests that have already reached the max model length.
                    continue

                valid_ngram_requests.append(i)

            return self.batch_propose(
                len(sampled_token_ids),
                valid_ngram_requests,
                num_tokens_no_spec,
                token_ids_cpu,
                num_speculative_tokens,
            )
