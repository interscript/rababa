# README


### Graph Parsed
> resources/TestFarsiTrans.*

### Python Dependencies
Has to be ran alongside the python dependancies of farsi branch

### Install Julia
(https://julialang.org/downloads/)[julia downloads]

### Julia Dependencies 
> julia packageInstall.jl 


### Commands
* Train
> julia train.jl --help

> julia train.jl --path-lucidchart-csv resources/TestFarsiTrans.csv --brain-entry transliterator  --path-model resources/TestFarsiTrans.dat

* Run 
> julia transliterateSingleString.jl --help

> julia transliterateSingleString.jl --path-model resources/TestFarsiTrans.dat --farsi-text یویو --pos-tagging N

