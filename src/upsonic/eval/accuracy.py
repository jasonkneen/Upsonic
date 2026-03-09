from __future__ import annotations
import asyncio
import time
from typing import Union, Optional, List, Dict, Any, TYPE_CHECKING

from rich.panel import Panel
from rich.text import Text

from upsonic.agent.agent import Agent
from upsonic.graph.graph import Graph
from upsonic.team.team import Team 
from upsonic.tasks.tasks import Task
from upsonic.eval.models import EvaluationScore, AccuracyEvaluationResult
from upsonic.eval._pl_helpers import extract_model_parameters, accumulate_agent_usage
from upsonic.utils.printing import console

if TYPE_CHECKING:
    from upsonic.integrations.promptlayer import PromptLayer


class AccuracyEvaluator:
    """
    The main orchestrator for running accuracy evaluations on Upsonic agents,
    graphs, or teams using the LLM-as-a-judge pattern.
    """

    def __init__(
        self,
        judge_agent: Agent,
        agent_under_test: Union[Agent, Graph, Team],
        query: str,
        expected_output: str,
        additional_guidelines: Optional[str] = None,
        num_iterations: int = 1,
        promptlayer: Optional["PromptLayer"] = None,
    ):
        if not isinstance(judge_agent, Agent):
            raise TypeError("The `judge_agent` must be an instance of the `Agent` agent class.")
        if not isinstance(agent_under_test, (Agent, Graph, Team)):
            raise TypeError("The `agent_under_test` must be an instance of `Agent`, `Graph`, or `Team`.")
        if not isinstance(num_iterations, int) or num_iterations < 1:
            raise ValueError("`num_iterations` must be a positive integer.")

        self.judge_agent: Agent = judge_agent
        self.agent_under_test: Union[Agent, Graph, Team] = agent_under_test
        self.query: str = query
        self.expected_output: str = expected_output
        self.additional_guidelines: str = additional_guidelines or "No additional guidelines provided."
        self.num_iterations: int = num_iterations
        self.promptlayer: Optional["PromptLayer"] = promptlayer
        self._results: List[EvaluationScore] = []

    async def run(self, print_results: bool = True) -> AccuracyEvaluationResult:
        self._results = []
        last_generated_output: str = ""
        eval_start_time: float = time.time()
        total_input_tokens: int = 0
        total_output_tokens: int = 0
        total_price: float = 0.0

        for i in range(self.num_iterations):
            if self.num_iterations > 1:
                console.print(f"[bold blue]--- Running Evaluation: Iteration {i + 1} of {self.num_iterations} ---[/bold blue]")

            generated_output_obj = None
            task = Task(description=self.query)

            if isinstance(self.agent_under_test, Agent):
                await self.agent_under_test.do_async(task)
                generated_output_obj = task.response
                in_tok, out_tok, cost = accumulate_agent_usage(self.agent_under_test)
                total_input_tokens += in_tok
                total_output_tokens += out_tok
                total_price += cost
            elif isinstance(self.agent_under_test, Graph):
                state = await self.agent_under_test.run_async(verbose=False)
                generated_output_obj = state.get_latest_output()
            elif isinstance(self.agent_under_test, Team):
                generated_output_obj = await asyncio.to_thread(self.agent_under_test.complete, task)
            
            if generated_output_obj is None:
                raise ValueError("The agent under test produced a None output, cannot proceed with evaluation.")
            
            last_generated_output = str(generated_output_obj)
            
            score_object: EvaluationScore = await self._get_judge_score(last_generated_output)
            in_tok_j, out_tok_j, cost_j = accumulate_agent_usage(self.judge_agent)
            total_input_tokens += in_tok_j
            total_output_tokens += out_tok_j
            total_price += cost_j

            self._results.append(score_object)
        
        eval_end_time: float = time.time()
        final_result: AccuracyEvaluationResult = self._aggregate_and_present_results(last_generated_output, print_results)

        if self.promptlayer is not None:
            await self._log_eval_to_promptlayer(
                final_result,
                start_time=eval_start_time,
                end_time=eval_end_time,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                price=total_price,
            )

        return final_result

    async def run_with_output(self, output: str, print_results: bool = True) -> AccuracyEvaluationResult:
        self._results = [] 
        eval_start_time: float = time.time()
        total_input_tokens: int = 0
        total_output_tokens: int = 0
        total_price: float = 0.0

        for i in range(self.num_iterations):
            if self.num_iterations > 1:
                console.print(f"[bold blue]--- Scoring Pre-existing Output: Iteration {i + 1} of {self.num_iterations} ---[/bold blue]")

            score_object: EvaluationScore = await self._get_judge_score(output)
            in_tok, out_tok, cost = accumulate_agent_usage(self.judge_agent)
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            total_price += cost

            self._results.append(score_object)

        eval_end_time: float = time.time()
        final_result: AccuracyEvaluationResult = self._aggregate_and_present_results(output, print_results)

        if self.promptlayer is not None:
            await self._log_eval_to_promptlayer(
                final_result,
                start_time=eval_start_time,
                end_time=eval_end_time,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                price=total_price,
            )

        return final_result

    async def _log_eval_to_promptlayer(
        self,
        final_result: AccuracyEvaluationResult,
        *,
        start_time: float,
        end_time: float,
        input_tokens: int,
        output_tokens: int,
        price: float,
    ) -> None:
        if self.promptlayer is None:
            return

        agent_model: str = getattr(self.agent_under_test, "model_name", "unknown")
        provider: str
        model: str
        provider, model = self.promptlayer._parse_provider_model(str(agent_model))

        avg_score_0_100: int = max(0, min(100, int(round(final_result.average_score * 10))))

        per_iteration_scores: dict[str, int] = {}
        for i, eval_score in enumerate(final_result.evaluation_scores):
            per_iteration_scores[f"iteration_{i}_score"] = max(0, min(100, int(round(eval_score.score * 10))))
            per_iteration_scores[f"iteration_{i}_passed"] = 100 if eval_score.is_met else 0

        all_met: bool = all(s.is_met for s in final_result.evaluation_scores)

        model_params: Optional[Dict[str, Any]] = None
        if isinstance(self.agent_under_test, Agent):
            model_params = extract_model_parameters(self.agent_under_test)

        await self.promptlayer.alog(
            provider=provider,
            model=model,
            input_text=final_result.user_query,
            output_text=final_result.generated_output,
            start_time=start_time,
            end_time=end_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            price=price,
            parameters=model_params,
            tags=["upsonic-eval", "accuracy-eval"],
            metadata={
                "eval_type": "accuracy",
                "expected_output": final_result.expected_output,
                "num_iterations": len(final_result.evaluation_scores),
                "average_score": final_result.average_score,
                "all_met": all_met,
            },
            score=avg_score_0_100,
            status="SUCCESS",
            function_name=f"accuracy_eval:{agent_model}",
            scores=per_iteration_scores,
        )

    async def _get_judge_score(self, generated_output: str) -> EvaluationScore:
        judge_prompt = self._construct_judge_prompt(generated_output)
        
        judge_task = Task(
            description=judge_prompt,
            response_format=EvaluationScore,
            not_main_task=True
        )
        await self.judge_agent.do_async(judge_task)
        score_object = judge_task.response

        if not isinstance(score_object, EvaluationScore):
            raise TypeError(f"Judge agent failed to return a valid EvaluationScore object. Received: {type(score_object)}")
            
        return score_object

    def _aggregate_and_present_results(
        self, 
        final_generated_output: str, 
        print_results: bool
    ) -> AccuracyEvaluationResult:
        if not self._results:
             raise RuntimeError("Evaluation finished without producing any results.")

        average_score = sum(score.score for score in self._results) / len(self._results)

        final_result = AccuracyEvaluationResult(
            evaluation_scores=self._results,
            average_score=average_score,
            user_query=self.query,
            expected_output=self.expected_output,
            generated_output=final_generated_output,
        )

        if print_results:
            self._print_formatted_results(final_result)

        return final_result

    def _construct_judge_prompt(self, generated_output: str) -> str:
        return f"""
        You are an impartial and meticulous AI evaluation expert. Your task is to analyze an AI agent's response based on a user's query, a ground-truth "expected" answer, and specific evaluation guidelines. You must provide a fair score from 1 to 10 and detailed, structured reasoning for your judgment.

        You MUST respond ONLY with a JSON object that strictly conforms to the `EvaluationScore` schema.

        ---
        **EVALUATION MATERIALS**
        ---

        <UserQuery>
        {self.query}
        </UserQuery>

        <ExpectedAnswer>
        This is the gold-standard, ideal answer.
        {self.expected_output}
        </ExpectedAnswer>

        <GeneratedAnswer>
        This is the answer produced by the agent you are evaluating.
        {generated_output}
        </GeneratedAnswer>

        <EvaluationGuidelines>
        Follow these specific rules when judging the generated answer.
        {self.additional_guidelines}
        </EvaluationGuidelines>
        """

    def _print_formatted_results(self, result: AccuracyEvaluationResult) -> None:
        last_score = result.evaluation_scores[-1]
        
        if result.average_score >= 8:
            color, title = "green", "[bold green]✅ Evaluation Passed[/bold green]"
        elif result.average_score >= 5:
            color, title = "yellow", "[bold yellow]⚠️ Evaluation Warning[/bold yellow]"
        else:
            color, title = "red", "[bold red]❌ Evaluation Failed[/bold red]"

        summary_text = Text()
        summary_text.append("Average Score: ", style="bold")
        summary_text.append(f"{result.average_score:.2f} / 10.0\n\n", style=f"bold {color}")
        summary_text.append("--- Last Run Details ---\n", style="dim")
        summary_text.append("User Query: ", style="bold")
        summary_text.append(f"{result.user_query}\n")
        summary_text.append("Generated Output: ", style="bold")
        summary_text.append(f"{result.generated_output}\n")
        summary_text.append("Judge's Reasoning: ", style="bold")
        summary_text.append(f"{last_score.reasoning}\n")
        summary_text.append("Judge's Critique: ", style="bold")
        summary_text.append(f"{last_score.critique}")

        console.print(Panel(summary_text, title=title, border_style=color, expand=False))
