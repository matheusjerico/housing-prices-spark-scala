name := "House Project"

version := "1.0"

scalaVersion := "2.11.12"

mainClass in (Compile, packageBin) := Some("HouseApp")

libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.4.5"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.2.0"
