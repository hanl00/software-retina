|Commit Name   	|Changes  	|Cython X   	|Loop   	|Issues     |Notes     |
|---	|---	|---	|---	|---    |---    |
|First commit	|Profiling, changed to memoryviews	|Cython is 1.5790419971055114x faster	|5	| None  |       |
|Replaced numpy functions  	|Reverted if checks to original, fixed imports, replaced numpy functions with owncode    	|Cython is 4.112961885739843x faster   	|10  	|Prange/nogil performance worse off, incorrect output values    |       |
|Remove nan occurences   	|Replaced some numpy calls to self defined functions, fixed output values   	|Cython is 4.2557936634626365x faster   	|10   	|None       |Profile :  400015 function calls in 0.691 seconds      |
|   	|   	|   	|   	|       |       |