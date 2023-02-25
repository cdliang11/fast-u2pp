# 
decoding_chunk_size=-1
num_decoding_left_chunks=-1
ctc_weight=0.3
reverse_weight=0.3

for x in {1..10}; do
    ./decode.sh \
    --decoding_chunk_size $decoding_chunk_size \
    --num_decoding_left_chunks $num_decoding_left_chunks \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --beam_size $x
done

decoding_chunk_size=-1
num_decoding_left_chunks=-1
ctc_weight=0.3
reverse_weight=0.5

for x in {1..10}; do
    ./decode.sh \
    --decoding_chunk_size $decoding_chunk_size \
    --num_decoding_left_chunks $num_decoding_left_chunks \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --beam_size $x
done

decoding_chunk_size=-1
num_decoding_left_chunks=-1
ctc_weight=0.3
reverse_weight=0.0

for x in {1..10}; do
    ./decode.sh \
    --decoding_chunk_size $decoding_chunk_size \
    --num_decoding_left_chunks $num_decoding_left_chunks \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --beam_size $x
done
