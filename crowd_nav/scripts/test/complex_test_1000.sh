for randome_seed in 5 1005 2005 3005 4005 5005 6005 7005 8005 9005;do
  python complex_test.py -m data/multi/LSTM_RL --human_num 5 --other_robot_num 2 --randomseed $randome_seed
  python complex_test.py -m data/multi/SARL --human_num 5 --other_robot_num 2 --randomseed $randome_seed
  python complex_test_st.py -m data/multi/st2_3frame --human_num 5 --other_robot_num 2 --randomseed $randome_seed
  python complex_test.py -m data/multi/HoR_DRL --human_num 5 --other_robot_num 2 --randomseed $randome_seed
#  python complex_test.py -m data/multi/HeR_DRL --human_num 5 --other_robot_num 2 --randomseed $randome_seed
done
