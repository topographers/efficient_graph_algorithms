mkdir -p "fgw_diffusion_results"

for NSAMPLES in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
do
  for SEED in 14323 84719 68053 70327 79561 20731 29221
  do
    for ARMIJO in True False
    do
      for DRF in True False
      do
        echo "n_samples ${NSAMPLES} seed ${SEED} Armijo ${ARMIJO} drf ${DRF}"
        python3 ./test_fgw_diffusion.py --seed $SEED --n_samples $NSAMPLES --use_armijo $ARMIJO --different_random_features $DRF > "fgw_diffusion_results/n${NSAMPLES}_seed${SEED}_armijo${ARMIJO}_drf${DRF}.txt"
      done
    done
  done
done
