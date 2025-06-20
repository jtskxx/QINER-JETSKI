#include <chrono>
#include <thread>
#include <mutex>
#include <cstdio>
#include <cstring>
#include <array>
#include <queue>
#include <atomic>
#include <vector>
#include <assert.h>
#ifdef _MSC_VER
#include <intrin.h>
#include <winsock2.h>
#pragma comment (lib, "ws2_32.lib")

#else
#include <signal.h>
#include <immintrin.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#endif

#include "K12AndKeyUtil.h"
#include "keyUtils.h"

constexpr unsigned long long POOL_VEC_SIZE = (((1ULL<<32) + 64)) >> 3;
constexpr unsigned long long POOL_VEC_PADDING_SIZE = (POOL_VEC_SIZE + 200 - 1) / 200 * 200;

static constexpr unsigned long long NUMBER_OF_INPUT_NEURONS = 0000;
static constexpr unsigned long long NUMBER_OF_OUTPUT_NEURONS = 0000;
static constexpr unsigned long long NUMBER_OF_TICKS = 0000;
static constexpr unsigned long long MAX_NEIGHBOR_NEURONS = 0000;
static constexpr unsigned long long NUMBER_OF_MUTATIONS = 0000;
static constexpr unsigned long long POPULATION_THRESHOLD = 0000;
static constexpr unsigned int SOLUTION_THRESHOLD = 0000;

void generateRandom2Pool(unsigned char miningSeed[32], unsigned char* pool)
{
    unsigned char state[200];
    memcpy(&state[0], miningSeed, 32);
    memset(&state[32], 0, sizeof(state) - 32);

    for (unsigned int i = 0; i < POOL_VEC_PADDING_SIZE; i += sizeof(state))
    {
        KeccakP1600_Permute_12rounds(state);
        memcpy(&pool[i], state, sizeof(state));
    }
}

void random2(
    unsigned char seed[32],
    const unsigned char* pool,
    unsigned char* output,
    unsigned long long outputSizeInByte)
{
    unsigned long long paddingOutputSize = (outputSizeInByte + 64 - 1) / 64;
    paddingOutputSize = paddingOutputSize * 64;
    std::vector<unsigned char> paddingOutputVec(paddingOutputSize);
    unsigned char* paddingOutput = paddingOutputVec.data();

    unsigned long long segments = paddingOutputSize / 64;
    unsigned int x[8] = { 0 };
    for (int i = 0; i < 8; i++)
    {
        x[i] = ((unsigned int*)seed)[i];
    }

    for (int j = 0; j < segments; j++)
    {
        for (int i = 0; i < 8; i++)
        {
            unsigned int base = (x[i] >> 3) >> 3;
            unsigned int m = x[i] & 63;

            unsigned long long u64_0 = ((unsigned long long*)pool)[base];
            unsigned long long u64_1 = ((unsigned long long*)pool)[base + 1];

            // Move 8 * 8 * j to the current segment. 8 * i to current 8 bytes element
            if (m == 0)
            {
                // some compiler doesn't work with bit shift 64
                *((unsigned long long*) & paddingOutput[j * 8 * 8 + i * 8]) = u64_0;
            }
            else
            {
                *((unsigned long long*) & paddingOutput[j * 8 * 8 + i * 8]) = (u64_0 >> m) | (u64_1 << (64 - m));
            }

            // Increase the positions in the pool for each element.
            x[i] = x[i] * 1664525 + 1013904223; // https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
        }
    }

    memcpy(output, paddingOutput, outputSizeInByte);
}

// Clamp the neuron value
template <typename T>
T clampNeuron(T neuronValue)
{
    if (neuronValue > 1)
    {
        return 1;
    }

    if (neuronValue < -1)
    {
        return -1;
    }
    return neuronValue;
}

void extract64Bits(unsigned long long number, char* output)
{
    int count = 0;
    for (int i = 0; i < 64; ++i)
    {
        output[i] = ((number >> i) & 1);
    }
}

template <
unsigned long long numberOfInputNeurons, // K
unsigned long long numberOfOutputNeurons,// L
unsigned long long numberOfTicks,        // N
unsigned long long numberOfNeighbors,    // 2M
unsigned long long populationThreshold,  // P
unsigned long long numberOfMutations,    // S
unsigned int solutionThreshold
>
struct Miner
{
    static constexpr unsigned long long numberOfNeurons = numberOfInputNeurons + numberOfOutputNeurons;
    static constexpr unsigned long long maxNumberOfNeurons = populationThreshold;
    static constexpr unsigned long long maxNumberOfSynapses = populationThreshold * numberOfNeighbors;
    static constexpr unsigned long long initNumberOfSynapses = numberOfNeurons * numberOfNeighbors;

    static_assert(numberOfInputNeurons % 64 == 0, "numberOfInputNeurons must be divided by 64");
    static_assert(numberOfOutputNeurons % 64 == 0, "numberOfOutputNeurons must be divided by 64");
    static_assert(maxNumberOfSynapses <= (0xFFFFFFFFFFFFFFFF << 1ULL), "maxNumberOfSynapses must less than or equal MAX_UINT64/2");
    static_assert(initNumberOfSynapses % 32 == 0, "initNumberOfSynapses must be divided by 32");
    static_assert(numberOfNeighbors % 2 == 0, "numberOfNeighbors must divided by 2");
    static_assert(populationThreshold > numberOfNeurons, "populationThreshold must be greater than numberOfNeurons");
    static_assert(numberOfNeurons > numberOfNeighbors, "Number of neurons must be greater than the number of neighbors");

    std::vector<unsigned char> poolVec;

    void initialize(unsigned char miningSeed[32])
    {
        // Init random2 pool with mining seed
        poolVec.resize(POOL_VEC_PADDING_SIZE);
        generateRandom2Pool(miningSeed, poolVec.data());
    }

    struct Synapse
    {
        char weight;
    };

    // Data for running the ANN
    struct Neuron
    {
        enum Type
        {
            kInput,
            kOutput,
            kEvolution,
        };
        Type type;
        char value;
        bool markForRemoval;
    };

    // Data for roll back
    struct ANN
    {
        Neuron neurons[maxNumberOfNeurons];
        Synapse synapses[maxNumberOfSynapses];
        unsigned long long population;
    };
    ANN bestANN;
    ANN currentANN;

    // Intermediate data
    struct InitValue
    {
        unsigned long long outputNeuronPositions[numberOfOutputNeurons];
        unsigned long long synapseWeight[initNumberOfSynapses / 32]; // each 64bits elements will decide value of 32 synapses
        unsigned long long synpaseMutation[numberOfMutations];
    } initValue;

    struct MiningData
    {
        unsigned long long inputNeuronRandomNumber[numberOfInputNeurons / 64 ];  // each bit will use for generate input neuron value
        unsigned long long outputNeuronRandomNumber[numberOfOutputNeurons / 64]; // each bit will use for generate expected output neuron value
    } miningData;

    unsigned long long neuronIndices[numberOfNeurons];
    char previousNeuronValue[maxNumberOfNeurons];

    unsigned long long outputNeuronIndices[numberOfOutputNeurons];
    char outputNeuronExpectedValue[numberOfOutputNeurons];

    long long neuronValueBuffer[maxNumberOfNeurons];

    void mutate(unsigned char nonce[32], int mutateStep)
    {
        // Mutation
        unsigned long long population = currentANN.population;
        unsigned long long synapseCount = population * numberOfNeighbors;
        Synapse* synapses = currentANN.synapses;

        // Randomly pick a synapse, randomly increase or decrease its weight by 1 or -1
        unsigned long long synapseMutation = initValue.synpaseMutation[mutateStep];
        unsigned long long synapseIdx = (synapseMutation >> 1) % synapseCount;
        // Randomly increase or decrease its value
        char weightChange = 0;
        if ((synapseMutation & 1ULL) == 0)
        {
            weightChange = -1;
        }
        else
        {
            weightChange = 1;
        }

        char newWeight = synapses[synapseIdx].weight + weightChange;

        // Valid weight. Update it
        if (newWeight >= -1 && newWeight <=1 )
        {
            synapses[synapseIdx].weight = newWeight;
        }
        else // Invalid weight. Insert a neuron
        {
            // Insert the neuron
            insertNeuron(synapseIdx);
        }

        // Clean the ANN
        while (scanRedundantNeurons() > 0)
        {
            cleanANN();
        }
    }

    // Get the pointer to all outgoing synapse of a neurons
    Synapse* getSynapses(unsigned long long neuronIndex)
    {
        return &currentANN.synapses[neuronIndex * numberOfNeighbors];
    }

    // Circulate the neuron index
    unsigned long long clampNeuronIndex(long long neuronIdx, long long value)
    {
        unsigned long long population = currentANN.population;
        long long nnIndex = 0;
        // Calculate the neuron index (ring structure)
        if (value >= 0)
        {
            nnIndex = neuronIdx + value;
        }
        else
        {
            nnIndex = neuronIdx + population + value;
        }
        nnIndex = nnIndex % population;
        return (unsigned long long)nnIndex;
    }

    // Remove a neuron and all synapses relate to it
    void removeNeuron(unsigned long long neuronIdx)
    {
        // Scan all its neigbor to remove their outgoing synapse point to the neuron
        for (long long neighborOffset = -(long long)numberOfNeighbors / 2; neighborOffset <= (long long)numberOfNeighbors / 2; neighborOffset++)
        {
            unsigned long long nnIdx = clampNeuronIndex(neuronIdx, neighborOffset);
            Synapse* pNNSynapses = getSynapses(nnIdx);

            long long synapseIndexOfNN = getIndexInSynapsesBuffer(nnIdx, -neighborOffset);
            if (synapseIndexOfNN < 0)
            {
                continue;
            }

            // The synapse array need to be shifted regard to the remove neuron
            if (synapseIndexOfNN >= numberOfNeighbors / 2)
            {
                for (long long k = synapseIndexOfNN; k < numberOfNeighbors - 1; ++k)
                {
                    pNNSynapses[k] = pNNSynapses[k + 1];
                }
                pNNSynapses[numberOfNeighbors - 1].weight = 0;
            }
            else
            {
                for (long long k = synapseIndexOfNN; k > 0; --k)
                {
                    pNNSynapses[k] = pNNSynapses[k - 1];
                }
                pNNSynapses[0].weight = 0;
            }
        }

        // Shift the synapse array and the neuron array
        for (unsigned long long shiftIdx = neuronIdx; shiftIdx < currentANN.population; shiftIdx++)
        {
            currentANN.neurons[shiftIdx] = currentANN.neurons[shiftIdx + 1];

            // Also shift the synapses
            memcpy(getSynapses(shiftIdx), getSynapses(shiftIdx + 1), numberOfNeighbors * sizeof(Synapse));
        }
        currentANN.population--;
    }

    unsigned long long getNeighborNeuronIndex(unsigned long long neuronIndex, unsigned long long neighborOffset)
    {
        unsigned long long nnIndex = 0;
        if (neighborOffset < (numberOfNeighbors / 2))
        {
            nnIndex = clampNeuronIndex(neuronIndex + neighborOffset, -(long long)numberOfNeighbors / 2);
        }
        else
        {
            nnIndex = clampNeuronIndex(neuronIndex + neighborOffset + 1, -(long long)numberOfNeighbors / 2);
        }
        return nnIndex;
    }

    void insertNeuron(unsigned long long synapseIdx)
    {
        unsigned long long incomingNeighborSynapseIdx = synapseIdx % numberOfNeighbors;
        unsigned long long outgoingNeuron = synapseIdx / numberOfNeighbors;

        Synapse* synapses = currentANN.synapses;
        Neuron* neurons = currentANN.neurons;
        unsigned long long& population = currentANN.population;

        // Copy original neuron to the inserted one and set it as  Neuron::kEvolution type
        Neuron insertNeuron;
        insertNeuron = neurons[outgoingNeuron];
        insertNeuron.type = Neuron::kEvolution;
        unsigned long long insertedNeuronIdx = outgoingNeuron + 1;

        char originalWeight = synapses[synapseIdx].weight;

        for (unsigned long long i = population; i > outgoingNeuron; --i)
        {
            neurons[i] = neurons[i - 1];

            // Also shift the synapses to the right
            memcpy(getSynapses(i), getSynapses(i - 1), numberOfNeighbors * sizeof(Synapse));
        }
        neurons[insertedNeuronIdx] = insertNeuron;
        population++;

        // Try to update the synapse of inserted neuron. All outgoing synapse is init as zero weight
        Synapse* pInsertNeuronSynapse = getSynapses(insertedNeuronIdx);
        for (unsigned long long synIdx = 0; synIdx < numberOfNeighbors; ++synIdx)
        {
            pInsertNeuronSynapse[synIdx].weight = 0;
        }

        // Copy the outgoing synapse of original neuron
        // Outgoing points to the left
        if (incomingNeighborSynapseIdx < numberOfNeighbors / 2)
        {
            if (incomingNeighborSynapseIdx > 0)
            {
                // Decrease by one because the new neuron is next to the original one
                pInsertNeuronSynapse[incomingNeighborSynapseIdx - 1].weight = originalWeight;
            }
            // Incase of the outgoing synapse point too far, don't add the synapse
        }
        else
        {
            // No need to adjust the added neuron but need to remove the synapse of the original neuron
            pInsertNeuronSynapse[incomingNeighborSynapseIdx].weight = originalWeight;
        }

        for (long long delta = -(long long)numberOfNeighbors / 2; delta <= (long long)numberOfNeighbors / 2; ++delta)
        {
            // Only process the neigbors
            if (delta == 0)
            {
                continue;
            }
            unsigned long long updatedNeuronIdx = clampNeuronIndex(insertedNeuronIdx, delta);

            long long insertedNeuronIdxInNeigborList = -1;
            for (long long k = 0 ; k < numberOfNeighbors; k++)
            {
                unsigned long long nnIndex = getNeighborNeuronIndex(updatedNeuronIdx, k);
                if (nnIndex == insertedNeuronIdx)
                {
                    insertedNeuronIdxInNeigborList = k;
                }
            }

            assert(insertedNeuronIdxInNeigborList >= 0);

            Synapse* pUpdatedSynapses = getSynapses(updatedNeuronIdx);
            if (delta < 0)
            {
                for (long long k = numberOfNeighbors - 1 ; k >= insertedNeuronIdxInNeigborList; --k)
                {
                    pUpdatedSynapses[k] = pUpdatedSynapses[k - 1];
                }

                // Incomming synapse from original neuron -> inserted neuron must be zero
                if (delta == -1)
                {
                    pUpdatedSynapses[insertedNeuronIdxInNeigborList].weight = 0;
                }
            }
            else // Case: neurons in range [inserted N4 N5 N6], left synapses will be affected
            {
                // Right side is kept as it is, only need to shift to the left side
                for (unsigned long long k = 0; k < insertedNeuronIdxInNeigborList; ++k)
                {
                    // Updated synapse
                    pUpdatedSynapses[k] = pUpdatedSynapses[k + 1];
                }
            }
        }
    }

    long long getIndexInSynapsesBuffer(unsigned long long neuronIdx, long long neighborOffset)
    {
        // Skip the case neuron point to itself and too far neighbor
        if (neighborOffset == 0
            || neighborOffset < -(long long)numberOfNeighbors / 2
            || neighborOffset > (long long)numberOfNeighbors / 2)
        {
            return -1;
        }

        long long synapseIdx = (long long)numberOfNeighbors / 2 + neighborOffset;
        if (neighborOffset >= 0)
        {
            synapseIdx = synapseIdx - 1;
        }

        return synapseIdx;
    }

    // Check which neurons/synapse need to be removed after mutation
    unsigned long long scanRedundantNeurons()
    {
        unsigned long long population = currentANN.population;
        Synapse* synapses = currentANN.synapses;
        Neuron* neurons = currentANN.neurons;

        unsigned long long numberOfRedundantNeurons = 0;
        for (unsigned long long i = 0; i < population; i++)
        {
            neurons[i].markForRemoval = false;
            if (neurons[i].type == Neuron::kEvolution)
            {
                bool allOutGoingZeros = true;
                bool allIncommingZeros = true;

                // Loop though its synapses for checkout outgoing synapses
                for (unsigned long long n = 0; n < numberOfNeighbors; n++)
                {
                    char synapseW = synapses[i * numberOfNeighbors + n].weight;
                    if (synapseW != 0)
                    {
                        allOutGoingZeros = false;
                        break;
                    }
                }

                // Loop through the neighbor neurons to check all incoming synapses
                for (long long neighborOffset = -(long long)numberOfNeighbors / 2; neighborOffset <= (long long)numberOfNeighbors / 2; neighborOffset++)
                {
                    unsigned long long nnIdx = clampNeuronIndex(i, neighborOffset);
                    Synapse* nnSynapses = getSynapses(nnIdx);

                    long long synapseIdx = getIndexInSynapsesBuffer(nnIdx, -neighborOffset);
                    if (synapseIdx < 0)
                    {
                        continue;
                    }
                    char synapseW = nnSynapses[synapseIdx].weight;

                    if (synapseW != 0)
                    {
                        allIncommingZeros = false;
                        break;
                    }
                }
                if (allOutGoingZeros || allIncommingZeros)
                {
                    neurons[i].markForRemoval = true;
                    numberOfRedundantNeurons++;
                }
            }
        }
        return numberOfRedundantNeurons;
    }

    // Remove neurons and synapses that do not affect the ANN
    void cleanANN()
    {
        Synapse* synapses = currentANN.synapses;
        Neuron* neurons = currentANN.neurons;
        unsigned long long& population = currentANN.population;

        // Scan and remove neurons/synapses
        unsigned long long neuronIdx = 0;
        while (neuronIdx < population)
        {
            if (neurons[neuronIdx].markForRemoval)
            {
                // Remove it from the neuron list. Overwrite data
                // Remove its synapses in the synapses array
                removeNeuron(neuronIdx);
            }
            else
            {
                neuronIdx++;
            }
        }
    }

    void processTick()
    {
        unsigned long long population = currentANN.population;
        Synapse* synapses = currentANN.synapses;
        Neuron* neurons = currentANN.neurons;

        // Memset value of current one
        memset(neuronValueBuffer, 0, sizeof(neuronValueBuffer));

        // Loop though all neurons
        for (long long n = 0; n < population; ++n)
        {
            const Synapse* kSynapses = getSynapses(n);
            long long neuronValue = neurons[n].value;
            // Scan through all neighbor neurons and sum all connected neurons.
            // The synapses are arranged as neuronIndex * numberOfNeighbors
            for (long long m = 0; m < numberOfNeighbors; m++)
            {
                char synapseWeight = kSynapses[m].weight;
                unsigned long long nnIndex = 0 ;
                if ( m < numberOfNeighbors / 2)
                {
                    nnIndex = clampNeuronIndex(n + m, -(long long)numberOfNeighbors / 2);
                }
                else
                {
                    nnIndex = clampNeuronIndex(n + m + 1, -(long long)numberOfNeighbors / 2);
                }

                // Weight-sum
                neuronValueBuffer[nnIndex] += synapseWeight * neuronValue;
            }
        }

        // Clamp the neuron value
        for (long long n = 0; n < population; ++n)
        {
            long long neuronValue = clampNeuron(neuronValueBuffer[n]);
            neurons[n].value = neuronValue;
        }
    }

    void runTickSimulation()
    {
        unsigned long long population = currentANN.population;
        Synapse* synapses = currentANN.synapses;
        Neuron* neurons = currentANN.neurons;

        // Save the neuron value for comparison
        for (unsigned long long i = 0 ; i < population; ++i)
        {
            // Backup the neuron value
            previousNeuronValue[i] = neurons[i].value;
        }

        for(unsigned long long tick = 0; tick < numberOfTicks; ++tick)
        {
            processTick();
            bool shouldExit = true;
            bool allNeuronsUnchanged = true;
            bool allOutputNeuronsIsNonZeros = true;
            for (long long n = 0; n < population; ++n)
            {
                // Neuron unchanged check
                if (previousNeuronValue[n] != neurons[n].value)
                {
                    allNeuronsUnchanged = false;
                }

                // Ouput neuron value check
                if (neurons[n].type == Neuron::kOutput && neurons[n].value == 0)
                {
                    allOutputNeuronsIsNonZeros = false;
                }
            }

            if (allOutputNeuronsIsNonZeros || allNeuronsUnchanged)
            {
                break;
            }

            // Copy the neuron value
            for (long long n = 0; n < population; ++n)
            {
                previousNeuronValue[n] = neurons[n].value;
            }
        }
    }

    unsigned int computeNonMatchingOutput()
    {
        unsigned long long population = currentANN.population;
        Neuron* neurons = currentANN.neurons;

        unsigned int R = 0;
        unsigned long long outputIdx = 0;
        for (unsigned long long i = 0; i < population; i++)
        {
            if (neurons[i].type == Neuron::kOutput)
            {
                if (neurons[i].value != outputNeuronExpectedValue[outputIdx])
                {
                    R++;
                }
                outputIdx++;
            }
        }
        return R;
    }

    void initInputNeuron()
    {
        unsigned long long population = currentANN.population;
        Neuron* neurons = currentANN.neurons;
        unsigned long long inputNeuronInitIndex = 0;

        char neuronArray[64] = { 0 };
        for (unsigned long long i = 0 ; i < population; ++i)
        {
            // Input will use the init value
            if (neurons[i].type == Neuron::kInput)
            {
                // Prepare new pack
                if (inputNeuronInitIndex % 64 == 0)
                {
                    extract64Bits(miningData.inputNeuronRandomNumber[inputNeuronInitIndex / 64], neuronArray);
                }
                char neuronValue = neuronArray[inputNeuronInitIndex % 64];

                // Convert value of neuron to trits (keeping 1 as 1, and changing 0 to -1.).
                neurons[i].value = (neuronValue == 0) ? -1 : neuronValue;

                inputNeuronInitIndex++;
            }
        }
    }

    void initOutputNeuron()
    {
        unsigned long long population = currentANN.population;
        Neuron* neurons = currentANN.neurons;
        for (unsigned long long i = 0 ; i < population; ++i)
        {
            if (neurons[i].type == Neuron::kOutput)
            {
                neurons[i].value = 0;
            }
        }
    }

    void initExpectedOutputNeuron()
    {
        char neuronArray[64] = { 0 };
        for (unsigned long long i = 0 ; i < numberOfOutputNeurons; ++i)
        {
            // Prepare new pack
            if (i % 64 == 0)
            {
                extract64Bits(miningData.outputNeuronRandomNumber[i / 64], neuronArray);
            }
            char neuronValue = neuronArray[i % 64];
            // Convert value of neuron (keeping 1 as 1, and changing 0 to -1.).
            outputNeuronExpectedValue[i] = (neuronValue == 0) ? -1 : neuronValue;
        }
    }

    unsigned int initializeANN(unsigned char* publicKey, unsigned char* nonce)
    {
        unsigned char hash[32];
        unsigned char combined[64];
        memcpy(combined, publicKey, 32);
        memcpy(combined + 32, nonce, 32);
        KangarooTwelve(combined, 64, hash, 32);

        unsigned long long& population = currentANN.population;
        Synapse* synapses = currentANN.synapses;
        Neuron* neurons = currentANN.neurons;

        // Initialization
        population = numberOfNeurons;

        // Initalize with nonce and public key
        random2(hash, poolVec.data(), (unsigned char*)&initValue, sizeof(InitValue));

        // Randomly choose the positions of neurons types
        for (unsigned long long i = 0 ; i < population; ++i)
        {
            neuronIndices[i] = i;
            neurons[i].type = Neuron::kInput;
        }
        unsigned long long neuronCount = population;
        for (unsigned long long i = 0 ; i < numberOfOutputNeurons; ++i)
        {
            unsigned long long outputNeuronIdx = initValue.outputNeuronPositions[i] % neuronCount;

            // Fill the neuron type
            neurons[neuronIndices[outputNeuronIdx]].type = Neuron::kOutput;
            outputNeuronIndices[i] = neuronIndices[outputNeuronIdx];

            neuronCount = neuronCount - 1;
            neuronIndices[outputNeuronIdx] = neuronIndices[neuronCount];
        }

        // Synapse weight initialization
        for (unsigned long long i = 0 ; i < (initNumberOfSynapses / 32); ++i)
        {
            const unsigned long long mask = 0b11;

            for (int j = 0; j < 32; ++j)
            {
                int shiftVal = j * 2;
                unsigned char extractValue = (unsigned char)((initValue.synapseWeight[i] >> shiftVal) & mask);
                switch (extractValue)
                {
                    case 2: synapses[32 * i + j].weight = -1; break;
                    case 3: synapses[32 * i + j].weight = 1; break;
                    default: synapses[32 * i + j].weight = 0;
                }
            }
        }

        // Init the neuron input and expected output value
        memcpy((unsigned char*)&miningData, poolVec.data(), sizeof(miningData));

        // Init input neuron value and output neuron
        initInputNeuron();
        initOutputNeuron();

        // Init expected output neuron
        initExpectedOutputNeuron();

        // Ticks simulation
        runTickSimulation();

        // Copy the state for rollback later
        memcpy(&bestANN, &currentANN, sizeof(ANN));

        // Compute R and roll back if neccessary
        unsigned int R = computeNonMatchingOutput();

        return R;
    }

    // Main function for verification
    bool findSolution(unsigned char* publicKey, unsigned char* nonce)
    {
        // Initialize
        unsigned int bestR = initializeANN(publicKey, nonce);

        for (unsigned long long s = 0; s < numberOfMutations; ++s)
        {
            // Do the mutation
            mutate(nonce, s);

            // Exit if the number of population reaches the maximum allowed
            if (currentANN.population >= populationThreshold)
            {
                break;
            }

            // Ticks simulation
            runTickSimulation();

            // Compute R and roll back if neccessary
            unsigned int R = computeNonMatchingOutput();
            if (R > bestR)
            {
                // Roll back
                memcpy(&currentANN, &bestANN, sizeof(bestANN));
            }
            else
            {
                bestR = R;

                // Better R. Save the state
                memcpy(&bestANN, &currentANN, sizeof(bestANN));
            }

            assert(bestANN.population <= populationThreshold);
        }

        // Check score
        unsigned int score = numberOfOutputNeurons - bestR;
        printf("Score: %u/%u (threshold: %u)\n", score, numberOfOutputNeurons, solutionThreshold);
        
        if (score >= solutionThreshold)
        {
            return true;
        }

        return false;
    }
};

static std::atomic<char> state(0);

static unsigned char computorPublicKey[32];
static unsigned char randomSeed[32];
static std::atomic<long long> numberOfMiningIterations(0);
static std::atomic<unsigned int> numberOfFoundSolutions(0);
static std::queue<std::array<unsigned char, 32>> foundNonce;
std::mutex foundNonceLock;

typedef Miner<NUMBER_OF_INPUT_NEURONS, NUMBER_OF_OUTPUT_NEURONS, NUMBER_OF_TICKS, MAX_NEIGHBOR_NEURONS, POPULATION_THRESHOLD, NUMBER_OF_MUTATIONS, SOLUTION_THRESHOLD> ActiveMiner;

static void hexToByte(const char* hex, uint8_t* byte, const int sizeInByte)
{
    for (int i = 0; i < sizeInByte; i++){
        sscanf(hex+i*2, "%2hhx", &byte[i]);
    }
}

int main(int argc, char* argv[])
{
    char miningID[128];
    char miningSeedHex[65];
    char nonceHex[65];
    
    printf("=== ANN Solution Verifier ===\n\n");
    
    // Get Mining ID
    printf("Enter Mining ID (Computor Public Identity): ");
    scanf("%s", miningID);
    
    // Get Mining Seed
    printf("Enter Mining Seed (32 bytes hex, 64 characters): ");
    scanf("%s", miningSeedHex);
    
    // Get Nonce
    printf("Enter Nonce (32 bytes hex, 64 characters): ");
    scanf("%s", nonceHex);
    
    // Validate hex string lengths
    if (strlen(miningSeedHex) != 64) {
        printf("Error: Mining Seed must be exactly 64 hex characters (32 bytes)\n");
        return 1;
    }
    if (strlen(nonceHex) != 64) {
        printf("Error: Nonce must be exactly 64 hex characters (32 bytes)\n");
        return 1;
    }

    // Convert from hex strings to byte arrays
    unsigned char miningSeed[32];
    unsigned char nonce[32];
    unsigned char computorPublicKey[32];

    hexToByte(miningSeedHex, miningSeed, 32);
    hexToByte(nonceHex, nonce, 32);
    getPublicKeyFromIdentity(miningID, computorPublicKey);

    printf("\nVerifying solution for:\n");
    printf("  MiningID: %s\n", miningID);
    printf("  Mining Seed: ");
    for(int i = 0; i < 32; i++) printf("%02x", miningSeed[i]);
    printf("\n");
    printf("  Nonce: ");
    for(int i = 0; i < 32; i++) printf("%02x", nonce[i]);
    printf("\n\n");

    // Create verifier and verify
    ActiveMiner miner;
    miner.initialize(miningSeed);
    
    bool isValid = miner.findSolution(computorPublicKey, nonce);
    
    if (isValid) {
        printf("\n✓ VALID SOLUTION\n");
    } else {
        printf("\n✗ INVALID SOLUTION\n");
    }

    printf("\nPress Enter to exit...");
    getchar(); // Clear buffer
    getchar(); // Wait for Enter

    return isValid ? 0 : 1;
}
