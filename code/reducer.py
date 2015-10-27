#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import logging
import sys
import numpy as np

if __name__ == "__main__":
    w_count =0
    w = np.zeros(4)
    #average all incoming w
    for line in sys.stdin:
        line = line.strip()
        w_count = w_count +1
        w = w+ np.array(map(float, line.split(' ')))
    w = w / w_count
				
    print(w)
				
	
							
    
