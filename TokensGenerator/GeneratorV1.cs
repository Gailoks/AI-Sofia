using System.Diagnostics.CodeAnalysis;

namespace TokensGenerator
{
    public class GeneratorV1
    {
        public IReadOnlyList<string> Generate(IEnumerable<string> samples, int limit)
        {
            var entries = new Dictionary<StirngSegment, long>(56249); //56249 is prime number
            var usedChars = new HashSet<char>();

            foreach (var sample in samples)
            {
                for (int i = 0; i < sample.Length; i++)
				{
                    var currentCharacter = sample[i];
                    if (char.IsLetter(currentCharacter) || char.IsDigit(currentCharacter))
                        usedChars.Add(currentCharacter);

                    for (int len = 2; len <= 5; i++)
                    {
                        var segment = new StirngSegment(i, len, sample);
                        if (segment.InRange() == false)
                            break;
                        Increment(entries, segment, len);
                    }
				}
            }

            var result = new List<string>(limit);

            foreach (var character in usedChars)
                result.Add(character.ToString());

            foreach (var preToken in entries.OrderByDescending(s => s.Value).Take(limit - result.Count).Select(s => s.Key))
                result.Add(preToken.EjectAsString());

            result.Sort((a, b) => a.Length - b.Length);
            return result;
        }

        private static void Increment<TKey>(Dictionary<TKey, long> dictionary, TKey key, int size = 1) where TKey : notnull
		{
			if (dictionary.TryGetValue(key, out var value))
				dictionary[key] = value + size;
			else dictionary.Add(key, size);
		}


        private struct StirngSegment
        {
            private readonly string _originalString;


            public int Offset { get; }

            public int Length { get; }


            public StirngSegment(int offset, int length, string originalString)
            {
                if (offset < 0)
                    throw new ArgumentOutOfRangeException(nameof(offset));
                if (length <= 0)
                    throw new ArgumentOutOfRangeException(nameof(length));

                Offset = offset;
                Length = length;
                _originalString = originalString;
            }

            public bool InRange()
            {
                return _originalString.Length - Length >= Offset;
            }

            public ReadOnlySpan<char> Eject()
            {
                return _originalString.AsSpan().Slice(Offset, Length);
            }
            
            public string EjectAsString()
            {
                return _originalString.Substring(Offset, Length);
            }

            public override int GetHashCode()
            {
                var span = Eject();
                if (Length == 1) return HashCode.Combine(span[0]);
                else if (Length == 2) return HashCode.Combine(span[0], span[1]);
                else if (Length == 3) return HashCode.Combine(span[0], span[1], span[2]);
                else if (Length == 4) return HashCode.Combine(span[0], span[1], span[2], span[3]);
                else if (Length == 5) return HashCode.Combine(span[0], span[1], span[2], span[3], span[4]);
                else if (Length == 6) return HashCode.Combine(span[0], span[1], span[2], span[3], span[4], span[5]);
                else if (Length == 7) return HashCode.Combine(span[0], span[1], span[2], span[3], span[4], span[5], span[6]);
                else return HashCode.Combine(span[0], span[1], span[2], span[3], span[4], span[5], span[6], span[7]);
            }

            //TODO: Optimize using unsafe memeby comporation
            public override bool Equals([NotNullWhen(true)] object? obj)
            {
                if (obj is StirngSegment ss && ss.Length == Length)
                {
                    var thisSegment = Eject();
                    var otherSegment = ss.Eject();

                    for (int i = 0; i < Length; i++)
                    {
                        if (thisSegment[i] != otherSegment[i])
                            return false;
                    }

                    return true;
                }
                else return false;
            }
        }
    }
}