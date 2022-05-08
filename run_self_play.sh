# python hrl_self.py --wandb-mode online --wandb-project vss-baguncinha-selfhrl --wandb-entity robocin --training-n-controlled-robots 1
# python hrl_self.py --wandb-name 1v1-pretraining --wandb-mode online --wandb-project vss-baguncinha-selfhrl --wandb-entity robocin --training-n-controlled-robots 1 --training-train-worker False --training-worker-checkpoint ./checkpoints/VSSHRL-v1/pretraining-worker 
python hrl_self.py --wandb-name 1v1-pretraining --wandb-mode online --wandb-project vss-baguncinha-selfhrl --wandb-entity robocin --training-n-controlled-robots 1 --training-worker-checkpoint ./checkpoints/VSSHRL-v1/pretraining-worker 

# TODO
# DEBUGAR VELOCIDADE
# SALVAR TUDO PARA REINICIAR (REPLAY BUFFER, OPTIMIZER, REDES)

# RODAR MAIS DE UM EP DE VALIDAÇÃO, DIMINUIR A FREQUENCIA E NÃO GRAVAR TODOS OS EPS
# AUMENTAR NOISE