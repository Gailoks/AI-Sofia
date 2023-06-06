using BenchmarkDotNet.Attributes;
using System.IO;
using System.Collections.Generic;
using TokensGenerator;

public class TokenizationTest
{
    private readonly List<string> _samples;



    public TokenizationTest()
    {
        _samples = new List<string>();
        foreach (var sample in Directory.EnumerateFiles("../../../../samples"))
        {
            var text = File.ReadAllText(sample);
            _samples.Add(text);
        }
    }


    [Benchmark]
    public void GenerateV0()
    {
        var tokensGenerator = new GeneratorV0();

        tokensGenerator.Generate(_samples, 500);
    }

    [Benchmark]
    public void GenerateV1()
    {
        var tokensGenerator = new GeneratorV1();

        tokensGenerator.Generate(_samples, 500);
    }
}