|Commit Name   	|Changes  	|Performance to run 100 sample loops    |Cython X (comparison against original sample no GPU)  	|Loop   	|Issues     |Notes/Profile     |
|---	|---	|---	|---    |---	|---    |---    |
|First commit	|Profiling, changed to memoryviews	|       |Cython is 1.5790419971055114x faster	|5	| None  |       |
|Replaced numpy functions  	|Reverted if checks to original, fixed imports, replaced numpy functions with owncode    	|      |Cython is 4.112961885739843x faster   	|10  	|Prange/nogil performance worse off, incorrect output values    |       |
|Remove nan occurences   	|Replaced some numpy calls to self defined functions, fixed output values   	|       |Cython is 4.2557936634626365x faster   	|10   	|None       |Profile :  400015 function calls in 0.691 seconds      |
|Removed masking   	|Deleted masking and calculation of F value (always about 1 rounded to the nearest integer)   	|       |Cython is 9.335271184466762x faster   	|10   	| None      |50015 function calls in 0.245 seconds       |
|Replaced native python multiply "*"      |Self defined python function to multiply and sum 2 arrays       |        |Cython is 10.69720190850668x faster       |10       | None      | 50015 function calls in 0.180 seconds       |
|Set datatype to int_16     |Changed  datatype to int_16, uint8 (too small and negative coefficients), changed input size from 1280 x 720 to 1920 x 1080     |      |Cython is 6.292198792209533x faster       |10       |Incorrect output on certain values (unrelated to removal of masking)       |50015 function calls in 0.437 seconds       |
|Prange with gil inside loop       |Implemented prange but couldnt remove python object calls (tried restructuring to 1d memory view)       |       |Cython is 13.734205595247653x faster       |10      |None       |250031 function calls in 0.153 seconds       |
|Prange nogil       |Remove gil requirements by converting to 3d memory view, added new testing method (previous method took account of retina initialisation with loading coefficients etc)      |         |Cython is 27.724704600059322x faster       |10       | None     |31 function calls in 0.049 seconds       |
|Test updates       |Reworked performance testing, comparisions vs original/original+GPU, added Killick's retina generator, updated gitignore     | 0.04177126884460449       |Cython is 50.73988588305842x faster   |10       |Incorrect output values vs origianl+GPU code    |No changes to the Cython code, unsure souce of performance improvements from 27x to 56x, profiling moved to 3rd column       |
|       |       |       |       |       |       |       |
|       |       |       |       |       |       |       |
|       |       |       |       |       |       |       |
|       |       |       |       |       |       |       |
|       |       |       |       |       |       |       |
|       |       |       |       |       |       |       |
|       |       |       |       |       |       |       |
