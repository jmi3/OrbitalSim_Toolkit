RK_COEFFICIENT_TABLES = {
    1: ([0], [[0]], [1]),
    
    2: ([0, 1/2], 
        [
            [0, 0], 
            [1/2, 0]
        ], 
        [0, 1]),
    
    3: ([0, 1/2, 1], 
        [
            [0, 0, 0], 
            [1/2, 0, 0], 
            [-1, 2, 0]
        ], 
        [1/6, 2/3, 1/6]),
    
    4: ([0, 1/2, 1/2, 1], 
        [
            [0, 0, 0, 0], 
            [1/2, 0, 0, 0], 
            [0, 1/2, 0, 0], 
            [0, 0, 1, 0]
        ], 
        [1/6, 1/3, 1/3, 1/6]),
        
    5: ([0, 1/5, 3/10, 4/5, 8/9, 1, 1],
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
        ],
        [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]),
 
    "5GPT": ([0, 1/4, 3/8, 12/13, 1, 1/2],
        [
            [0, 0, 0, 0, 0, 0],
            [1/4, 0, 0, 0, 0, 0],
            [3/32, 9/32, 0, 0, 0, 0],
            [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
            [439/216, -8, 3680/513, -845/4104, 0, 0],
            [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0]
        ],
        [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]),

    6: ([0, 1/3, 2/3, 1/3, 1/2, 1/2, 1],
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1/3, 0, 0, 0, 0, 0, 0],
            [0, 2/3, 0, 0, 0, 0, 0],
            [1/12, 1/3, -1/12, 0, 0, 0, 0],
            [-1/16, 9/8, -3/16, -3/8, 0, 0, 0],
            [0, 9/8, -3/8, -3/4, 1/2, 0, 0],
            [9/44, -9/11, 63/44, 18/11, 0, -16/11, 0],
        ],
        [11/120, 0, 27/40, 27/40, -4/15, -4/15, 11/120]),

    "6GPT": ([0, 1/4, 1/4, 1/2, 3/4, 1, 1],
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1/4, 0, 0, 0, 0, 0, 0],
            [1/8, 1/8, 0, 0, 0, 0, 0],
            [0, 0, 1/2, 0, 0, 0, 0],
            [3/16, -3/8, 3/8, 9/16, 0, 0, 0],
            [-3/7, 8/7, 6/7, -12/7, 8/7, 0, 0],
            [7/90, 0, 32/90, 12/90, 32/90, 7/90, 0]
        ],
        [7/90, 0, 32/90, 12/90, 32/90, 7/90, 0])
}