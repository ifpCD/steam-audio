//
// Copyright 2017-2023 Valve Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

using UnityEngine;

namespace SteamAudio
{
    internal static class Sampling
    {
        // Sampling::generateSphereVolumeSamples
        public static void GenerateSphereVolumeSamples(int numSamples, UnityEngine.Vector3[] samples)
        {
            for (int i = 0; i < numSamples; ++i)
            {
                float uPhi = RadicalInverse(2, i);
                float uTheta = RadicalInverse(3, i);
                float uR = RadicalInverse(5, i);

                float phi = 2.0f * Mathf.PI * uPhi;
                float theta = Mathf.Acos(2.0f * uTheta - 1.0f);

                float r = Mathf.Pow(uR, 1.0f / 3.0f);

                float x = r * Mathf.Sin(theta) * Mathf.Cos(phi);
                float y = r * Mathf.Sin(theta) * Mathf.Sin(phi);
                float z = r * Mathf.Cos(theta);

                // axis swap
                samples[i] = new UnityEngine.Vector3(x, y, -z);
            }
        }

        // Sampling::radicalInverse
        public static float RadicalInverse(int p, int i)
        {
            float inv = 1.0f / p;
            int reversed = 0;
            float invN = 1.0f;

            while (i > 0)
            {
                int next = i / p;
                int digit = i - (next * p);
                reversed = reversed * p + digit;
                invN *= inv;
                i = next;
            }

            return Mathf.Min(reversed * invN, 1.0f);
        }

        // Sampling::transformSphereVolumeSample (flat parameter version)
        public static UnityEngine.Vector3 TransformSphereVolumeSample(UnityEngine.Vector3 sample, UnityEngine.Vector3 center, float radius)
        {
            return center + (sample * radius);
        }
    }
}
