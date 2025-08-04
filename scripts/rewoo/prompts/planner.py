DEFAULT_PREFIX = "For the following tasks, make plans that can solve the problem step-by-step. For each plan, " \
                 "indicate which external tool together with tool input to retrieve evidence. You can store the " \
                 "evidence into a variable #E that can be called by later tools. (Plan, #E1 = , Plan, #E2 = , Plan, ...) \n\n"
# DEFAULT_SUFFIX = "Begin! Describe your plans with rich details. Each Plan should be followed by only one #E. \n\n"
DEFAULT_SUFFIX = (
    "Begin! For each task, describe your reasoning step-by-step using structured plans. "
    "⚠️ Each Plan **must** be immediately followed by exactly one evidence variable (#E). "
    "Do not generate a #E without a preceding Plan. Do not skip a Plan before generating a #E.\n\n"
    "Repeat this format until the question can be answered. Avoid extra or missing steps.\n\n"
)

DEFAULT_FEWSHOT = "\n"

RESOURCE_RELUCTANT_SUFFIX = "Begin! Make as few plans as possible if it can solve the problem. Each Plan should be followed by only one #E. \n\n"