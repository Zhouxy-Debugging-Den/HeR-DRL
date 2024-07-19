for randome_seed in 5 1005 2005 3005 4005 5005 6005 7005 8005 9005;do
  python complex_test.py -m data/multi/HoR_DRL_nocate --human_num 5 --other_robot_num 2 --randomseed $randome_seed
  python complex_test.py -m data/multi/HeR_DRL_nocate --human_num 5 --other_robot_num 2 --randomseed $randome_seed
done
