export HUGGINGFACE_TOKEN=""
python -c "from huggingface_hub import login; login('$HUGGINGFACE_TOKEN')"
export METHOD=rewoo
export MODEL=agent-llama
export LM=$MODEL

go() {
    python run_eval.py \
    --method $METHOD \
    --dataset $TASK \
    --sample_size 50 \
    --toolset ${TOOL[@]} \
    --base_lm $LM \
    --save_result > >(tee tee results/eval_${TASK}_${METHOD}_${MODEL}.log) 2> >(tee results/eval_${TASK}_${METHOD}_${MODEL}.err)
}


export TASK=hotpot_qa
export TOOL=(Wikipedia LLM)
go

export TASK=trivia_qa
export TOOL=(Wikipedia LLM)
go

