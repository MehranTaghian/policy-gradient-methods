tmux new-session -d -s ppo0 "python _ppo.py 0"
tmux new-session -d -s ppo1 "python _ppo.py 1"
tmux new-session -d -s ppo2 "python _ppo.py 2"
tmux new-session -d -s ppo3 "python _ppo.py 3"
tmux new-session -d -s ppo4 "python _ppo.py 4"
tmux new-session -d -s ppo5 "python _ppo.py 5"
tmux new-session -d -s ppo6 "python _ppo.py 6"
tmux new-session -d -s ppo7 "python _ppo.py 7"
tmux new-session -d -s ppo8 "python _ppo.py 8"
tmux new-session -d -s ppo9 "python _ppo.py 9"

tmux new-session -d -s reinforce0 "python _reinforce.py 0"
tmux new-session -d -s reinforce1 "python _reinforce.py 1"
tmux new-session -d -s reinforce2 "python _reinforce.py 2"
tmux new-session -d -s reinforce3 "python _reinforce.py 3"
tmux new-session -d -s reinforce4 "python _reinforce.py 4"
tmux new-session -d -s reinforce5 "python _reinforce.py 5"
tmux new-session -d -s reinforce6 "python _reinforce.py 6"
tmux new-session -d -s reinforce7 "python _reinforce.py 7"
tmux new-session -d -s reinforce8 "python _reinforce.py 8"
tmux new-session -d -s reinforce9 "python _reinforce.py 9"

python _reinforce.py 0
python _reinforce.py 1
python _reinforce.py 2
python _reinforce.py 3
python _reinforce.py 4
python _reinforce.py 5
python _reinforce.py 6
python _reinforce.py 7
python _reinforce.py 8
python _reinforce.py 9


tmux new-session -d -s ppo0 "python new_ppo.py 0"
tmux new-session -d -s ppo1 "python new_ppo.py 1"
tmux new-session -d -s ppo2 "python new_ppo.py 2"
tmux new-session -d -s ppo3 "python new_ppo.py 3"
tmux new-session -d -s ppo4 "python new_ppo.py 4"
tmux new-session -d -s ppo5 "python new_ppo.py 5"
tmux new-session -d -s ppo6 "python new_ppo.py 6"
tmux new-session -d -s ppo7 "python new_ppo.py 7"
tmux new-session -d -s ppo8 "python new_ppo.py 8"
tmux new-session -d -s ppo9 "python new_ppo.py 9"

python new_ppo.py 0
python new_ppo.py 1
python new_ppo.py 2
python new_ppo.py 3
python new_ppo.py 4
python new_ppo.py 5
python new_ppo.py 6
python new_ppo.py 7
python new_ppo.py 8
python new_ppo.py 9

