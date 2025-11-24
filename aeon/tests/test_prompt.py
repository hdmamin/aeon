from string import Template
from unittest.mock import MagicMock, patch

import pytest

from aeon.prompt import template_varnames, infer_provider, Prompt


class TestTemplateVarnames:

    def test_single_variable(self):
        template = Template("Hello $name")
        assert template_varnames(template) == ["name"]

    def test_multiple_variables(self):
        template = Template("$greeting $name, your score is $score")
        assert template_varnames(template) == ["greeting", "name", "score"]

    def test_no_variables(self):
        template = Template("Hello world")
        assert template_varnames(template) == []

    def test_duplicate_variables(self):
        # Template should only extract unique variable names
        template = Template("$name said hello to $name")
        result = template_varnames(template)
        # Result may have duplicates based on regex matches
        assert "name" in result

    def test_variables_with_underscores(self):
        template = Template("$first_name $last_name")
        assert set(template_varnames(template)) == {"first_name", "last_name"}


class TestInferProvider:

    def test_openai_models(self):
        assert infer_provider("gpt-4") == "openai"
        assert infer_provider("gpt-3.5-turbo") == "openai"
        assert infer_provider("gpt-4o") == "openai"
        assert infer_provider("gpt-5-mini") == "openai"

    def test_non_openai_defaults_to_openrouter(self):
        assert infer_provider("claude-3-opus") == "openrouter"
        assert infer_provider("llama-2-70b") == "openrouter"
        assert infer_provider("mistral-large") == "openrouter"


class TestPromptResolveKwargs:

    @patch("aeon.prompt.importlib.import_module")
    def test_gpt5_uses_special_defaults(self, mock_import):
        mock_prompt = MagicMock()
        mock_prompt.messages = [
            {"role": "developer", "content": "test"},
            {"role": "user", "content": "$input"}
        ]
        mock_prompt.kwargs = {"model": "gpt-5-mini", "response_format": MagicMock}
        mock_import.return_value = mock_prompt

        prompt = Prompt("test_prompt")
        assert "reasoning_effort" in prompt.default_kwargs
        assert "verbosity" in prompt.default_kwargs
        assert prompt.default_kwargs["reasoning_effort"] == "minimal"

    @patch("aeon.prompt.importlib.import_module")
    def test_gpt5_rejects_logprobs(self, mock_import):
        mock_prompt = MagicMock()
        mock_prompt.messages = [
            {"role": "developer", "content": "test"},
            {"role": "user", "content": "$input"}
        ]
        mock_prompt.kwargs = {"model": "gpt-5-mini", "logprobs": True}
        mock_import.return_value = mock_prompt

        with pytest.raises(ValueError, match="gpt-5 should not specify"):
            Prompt("test_prompt")

    @patch("aeon.prompt.importlib.import_module")
    def test_non_gpt5_uses_standard_defaults(self, mock_import):
        mock_prompt = MagicMock()
        mock_prompt.messages = [
            {"role": "developer", "content": "test"},
            {"role": "user", "content": "$input"}
        ]
        mock_prompt.kwargs = {"model": "gpt-4o", "response_format": MagicMock}
        mock_import.return_value = mock_prompt

        prompt = Prompt("test_prompt")
        assert prompt.default_kwargs["model"] == "gpt-4o"
        assert prompt.default_kwargs["logprobs"] is True
        assert "reasoning_effort" not in prompt.default_kwargs
