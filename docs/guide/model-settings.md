# Model settings

Pydantic AI model settings are sent with every request, the screen check's too. A local model that
thinks before answering, for one, can be told not to:

```bash
framelift run samples --profile examples.fantasy_shop:profile --model openai-chat:<model> \
  --out out --model-setting openai_reasoning_effort=none
```

(`Pipeline(..., model_settings={"openai_reasoning_effort": "none"})` in Python, `framelift eval
--model-setting` for evaluations.) `--model-setting KEY=VALUE` can be repeated; VALUE is read as
JSON, or else as a string, so `temperature=0` is a number and `openai_reasoning_effort=none` a
string; quote a string that would read as JSON (`openai_user='"12345"'`). Settings prefixed with
another provider's name, such as `anthropic_`, are ignored, so a `screen_model` of another
provider skips them. Settings without a prefix, such as `temperature` or `max_tokens`, go to the
screen check and the extraction alike. Keys are not checked: a misspelled one is ignored too.
