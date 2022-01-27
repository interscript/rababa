import Pkg

meta = ["Genie","PyCall","Distributions","YAML","HTTP","JSON","Dates","Logging","Serialization","ProgressMeter","SparseArrays","IJulia","CSV","StringDistances","GraphPlot","Graphs","Colors","Cairo","Compose","Base64"]


Pkg.update()

for p in meta
    Pkg.add(p)
end

