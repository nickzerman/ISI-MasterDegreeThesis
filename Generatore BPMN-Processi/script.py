exec(open("utils.py").read())
exec(open("generate_impacts.py").read())
exec(open("sese_diagram.py").read())
exec(open("stats.py").read())
exec(open("utils.py").read())
exec(open("algorithms.py").read())

import time
import pandas as pd
r =[]
limit = (-1,-1,-1,-1)
for i in range(1,11):
    for j in range(1, 11):
        for l in range(1, 11):
            for d in range(1, 11):
                if (i > limit[0] 
                    or (i == limit[0] and j > limit[1]) 
                    or (i == limit[0] and j == limit[1] and l > limit[2])
                    or (i == limit[0] and j == limit[1] and l == limit[2] and d > limit[3])
                    ):
                    for mode in ["random", "bagging_divide", "bagging_remove","bagging_remove_divide","bagging_remove_reverse", "bagging_remove_reverse_divide"]:
                        p = get_process_from_file(i,j,l)
                        n = find_max_task_number(p)
                        v = generate_vectors(n,d, mode=mode)
                        imp = dict(zip([f"T{i}" for i in range(1,n+1)], v))
                        t = PARSER.parse(p)
                        start_time = time.time()
                        build_max_values(t, imp)
                        end_time = time.time()
                        max_time = end_time - start_time
                        start_time = time.time()
                        build_pareto_frontier(t, imp)
                        end_time = time.time()
                        pareto_time = end_time - start_time
                        r.append({ 
                        "nested": i, 
                        "independent": j, 
                        "process_number": l,
                        "dimensions": d,
                        "mode": mode, 
                        "max_pareto": max_pareto_length(t), 
                        "max_theoretical_pareto": max_theoretical_pareto_length(t),
                        "max_time": max_time,
                        "pareto_time": pareto_time
                        }
                        )
                        print(r[-1])
                        pd.DataFrame(r).to_csv("results.csv",index=False)