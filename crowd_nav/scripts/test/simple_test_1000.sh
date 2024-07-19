for human_num in 5;do
  for randome_seed in 5 1005 2005 3005 4005 5005 6005 7005 8005 9005;do
    python simple_test.py -m data/single/LSTM_RL --human_num $human_num --randomseed $randome_seed
    python simple_test.py -m data/single/SARL --human_num $human_num --randomseed $randome_seed
    python simple_test.py -m data/single/HoR_DRL --human_num $human_num --randomseed $randome_seed
    python simple_test.py -m data/single/HeR_DRL --human_num $human_num --randomseed $randome_seed
    python simple_test.py -m data/single/GAT_DRL --human_num $human_num --randomseed $randome_seed
    python simple_test_st.py -m data/single/st2_3frame --human_num $human_num --randomseed $randome_seed
  done
done
