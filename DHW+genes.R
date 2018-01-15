W = read.table("~/WFinal.csv", sep=",", header =F)

Gene_list = read.table("~/InitialGenes13k.csv", sep=",", header =T, stringsAsFactors = F)
rownames(W) <- Gene_list$Gene

tf <- function(x) {
  sx <- scale(x)
  ind <- which(abs(sx) > 3.94)
  return(Gene_list$Gene[ind])
}

HWG = apply(sampleweights, 2, tf)
DHW = as.data.frame(table(unlist(HWG)))

cat(as.character(DHW$Var1), sep = "\n")
