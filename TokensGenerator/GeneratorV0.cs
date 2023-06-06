using System.Diagnostics.CodeAnalysis;
using System.Text;

namespace TokensGenerator
{
	public class GeneratorV0
	{
		public IReadOnlyList<string> Generate(IEnumerable<string> samples, int limit)
		{
			var dictionary = new List<string>(limit);

			var entries = new Dictionary<string, long>();
			var usedChars = new HashSet<char>();
			long totalChars = 0;

			foreach (var sample in samples)
			{
				for (int i = 0; i < sample.Length; i++)
				{
					totalChars++;

					var currentCharacter = sample[i];
					char? prev1Character = i >= 1 ? sample[i - 1] : null;
					char? prev2Character = i >= 2 ? sample[i - 2] : null;
					char? prev3Character = i >= 3 ? sample[i - 3] : null;
					char? prev4Character = i >= 4 ? sample[i - 4] : null;

    				if (char.IsLetter(currentCharacter) || char.IsDigit(currentCharacter))
						usedChars.Add(currentCharacter);

					if (prev1Character is not null)
	    			{
						Increment(entries, $"{prev1Character.Value}{currentCharacter}", size: 2);
						if (prev2Character is not null)
						{
							Increment(entries, $"{prev2Character.Value}{prev1Character.Value}{currentCharacter}", size: 3);
							if (prev3Character is not null)
							{
								Increment(entries, $"{prev3Character.Value}{prev2Character.Value}{prev1Character.Value}{currentCharacter}", size: 4);
								if (prev4Character is not null)
									Increment(entries, $"{prev4Character.Value}{prev3Character.Value}{prev2Character.Value}{prev1Character.Value}{currentCharacter}", size: 5);
							}
						}
					}
				}
			}


			foreach (var character in usedChars)
				dictionary.Add(character.ToString());

			foreach (var preToken in entries.OrderByDescending(s => s.Value).Take(limit - dictionary.Count).Select(s => s.Key))
				dictionary.Add(preToken);

			dictionary.Sort((a, b) => a.Length - b.Length);
			return dictionary;
		}

		private static void Increment<TKey>(Dictionary<TKey, long> dictionary, TKey key, int size = 1) where TKey : notnull
		{
			if (dictionary.TryGetValue(key, out var value))
				dictionary[key] = value + size;
			else dictionary.Add(key, size);
		}
	}
}