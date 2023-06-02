class Prompter:

    SUMMARIZE_PROMPT = "Write an excellent summary of the given text:\n\nTitle: {title}\n\nText: {post}\n\nTL;DR: {summary}"
    FEEDBACK_PROMPT = "Given a summary of the given text, write feedback on the summary:\n\nTitle: {title}\n\nText: {post}\n\nTL;DR: {summary}\n\nFeedback: {feedback}"
    REFINEMENT_FEEDBACK_PROMPT = "Given a summary of the given text, write feedback on the summary. After that, write an excellent summary that incorporates the feedback on the given summary and is better than the given summary:\n\nTitle: {title}\n\nText: {post}\n\nTL;DR: {summary}\n\nFeedback on summary: {feedback}\n\nImproved TL;DR: {refinement}"
    REFINEMET_PROMPT = "Write an excellent summary that incorporates the feedback on the given summary and is better than the given summary.\n\nTitle: {title}\n\nText: {post}\n\nTL;DR: {summary}\n\nFeedback on summary: {feedback}\n\nImproved TL;DR: {refinement}"

    def __init__(self, prompt_type: str) -> None:
        assert prompt_type in [
            "summarize",
            "generate_feedback",
            "generate_feedback_refinement",
            "generate_refinement",
        ], "The given prompt type is not correct"
        self.prompt_type = prompt_type

    def _generate_summary_prompt_format(
        self, title: str, post: str, summary: str
    ) -> str:
        return (
            self.SUMMARIZE_PROMPT.format(title=title, post=post, summary=summary),
            "TL;DR: ",
        )

    def _generate_feedback_prompt_format(
        self, title: str, post: str, summary: str, feedback: str
    ) -> str:
        return (
            self.FEEDBACK_PROMPT.format(
                title=title, post=post, summary=summary, feedback=feedback
            ),
            "Feedback: ",
        )

    def _generate_feedback_refinement_prompt_format(
        self, title: str, post: str, summary: str, feedback: str, refinement: str
    ) -> str:
        return (
            self.REFINEMENT_FEEDBACK_PROMPT.format(
                title=title,
                post=post,
                summary=summary,
                feedback=feedback,
                refinement=refinement,
            ),
            "Feedback on summary: ",
        )

    def _generate_refinement_prompt_format(
        self, title: str, post: str, summary: str, feedback: str
    ) -> str:
        return (
            self.REFINEMENT_PROMPT.format(
                title=tiel,
                post=post,
                summary=summary,
                feedback=feedback,
                refinement=refinement,
            ),
            "Improved TL;DR: ",
        )

    def get_prompt_and_completion(
        self,
        title: str,
        post: str,
        summary: str = None,
        feedback: str = None,
        refinement: str = None,
    ) -> str:

        if self.prompt_type == "summarize":
            return _generate_summary_prompt_format(title, post, summary)
        elif self.prompt_type == "generate_feedback":
            return _generate_feedback_prompt_format(title, post, summary, feedback)
        elif self.prompt_type == "generate_feedback_refinement":
            return _generate_feedback_refinement_prompt_format(
                title, post, summary, feedback, refinement
            )
        elif self.prompt_type == "generate_refinement":
            return _generate_refinement_prompt_format(
                title, post, summary, feedback, refinement
            )
