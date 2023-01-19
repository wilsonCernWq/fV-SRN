#pragma once

#include <iostream>

struct LZ4Compressor_Impl;
class LZ4Compressor
{
public:
	static constexpr int FAST_COMPRESSION = 0;
	static constexpr int MIN_COMPRESSION = 3;
	static constexpr int MAX_COMPRESSION = 9;
	static constexpr int MAX_CHUNK_SIZE = 64 * 1024;

	/**
	 * \brief Creates the compressor.
	 * The level can be either FAST_COMPRESSION or
	 * a value between MIN_COMPRESSION and MAX_COMPRESSION (inclusive).
	 */
	explicit LZ4Compressor(int level = MAX_COMPRESSION);
	~LZ4Compressor();

	LZ4Compressor(const LZ4Compressor& other) = delete;
	LZ4Compressor(LZ4Compressor&& other) noexcept = default;
	LZ4Compressor& operator=(const LZ4Compressor& other) = delete;
	LZ4Compressor& operator=(LZ4Compressor&& other) noexcept = default;
	
	/**
	 * \brief Compresses a block of memory and writes the result to the output stream 'dst'.
	 * The maximal size of 'src' is specified by MAX_CHUNK_SIZE.
	 *
	 * Note: It is assumed that the src memory from previous invocations
	 * remain present, unmodified and at the same address in memory!
	 * 
	 * \param dst the output stream
	 * \param src the input chunk of memory
	 * \param srcSize the size of src
	 */
	void compress(std::ostream& dst, const char* src, int srcSize);

private:
	LZ4Compressor_Impl* impl_;
};

struct LZ4Decompressor_Impl;
class LZ4Decompressor
{
public:
	LZ4Decompressor();
	~LZ4Decompressor();

	LZ4Decompressor(const LZ4Decompressor& other) = delete;
	LZ4Decompressor(LZ4Decompressor&& other) noexcept = default;
	LZ4Decompressor& operator=(const LZ4Decompressor& other) = delete;
	LZ4Decompressor& operator=(LZ4Decompressor&& other) noexcept = default;

	/**
	 * \brief Decompresses a chunk of memory.
	 * The compressed chunk is read from the input stream 'src',
	 * decompressed and placed in 'dst'.
	 * If the capacity of dst is not sufficient, a runtime_error is thrown.
	 *
	 * This function is guaranteed to never write outside of dst[0] to dst[dstCapacity-1].
	 * Note: The last 64KB of the decoded data (in 'dst') must remain available,
	 * unmodified and at the same memory position.
	 * 
	 * \param dst the destination memory
	 * \param dstCapacity the available free memory in 'dst'.
	 * \param src the source stream
	 * \return the number of bytes written to 'dst'
	 */
	int decompress(char* dst, int dstCapacity, std::istream& src);

private:
	LZ4Decompressor_Impl* impl_;
};
