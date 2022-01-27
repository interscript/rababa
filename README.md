# README

## Farsi Learn

Farsi learn is a software allowing a non technical experts to 
sketch rules and logic steps needed for the task of transliteration of Farsi.

We have created this system to speed up and make flexible this general part of the work. 

Whie the language specialist can design and try easily his own rules, the developer implements the code corresponding to the rules. 
This way both tasks and roles are well defined and confined.

### Graph Parsed
> resources/TestFarsiTrans.*

### Python Dependencies
Has to be ran alongside the python dependancies of farsi branch
> ENV["PYTHON"] = "...my python path with hazm installed..."
> import Pkg
> Pkg.add("PyCall")
> Pkg.build("PyCall") 

### Install Julia
[julia downloads](https://julialang.org/downloads/)

### Julia Dependencies 
> julia packageInstall.jl 


### Commands
* Train
> julia train.jl --help

> julia train.jl --path-lucidchart-csv resources/TestFarsiTrans.csv --brain-entry transliterator  --path-model resources/TestFarsiTrans.dat

* Run 
> julia transliterateSingleString.jl --help

> julia transliterateSingleString.jl --path-model resources/TestFarsiTrans.dat --farsi-text یویو --pos-tagging N

