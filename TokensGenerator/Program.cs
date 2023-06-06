using TokensGenerator;

var samples = new List<string>();
foreach (var sample in Directory.EnumerateFiles("samples"))
{
    var text = File.ReadAllText(sample);
    samples.Add(text);
}

var generatorV0 = new GeneratorV0();
var generatorV1 = new GeneratorV1();

var tokensV0 = generatorV0.Generate(samples, 500);
var tokensV1 = generatorV1.Generate(samples, 500);

Console.ForegroundColor = ConsoleColor.White;

Console.WriteLine($"Len of V0: {tokensV0.Count}, Len of V1: {tokensV1.Count}");

for (int i = 0; i < 500; i++)
{
    Console.ForegroundColor = tokensV0[i] == tokensV1[i] ? ConsoleColor.Green : ConsoleColor.Red;
    Console.WriteLine($"[{i}]:\t{tokensV0[i]}\t<->\t{tokensV1[i]}");
}