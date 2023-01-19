#include "lz4cpp.hpp"

extern "C"
{
#include "lz4.h"
#include "lz4hc.h"
}

#include <vector>
#include <string>

struct LZ4Compressor_Impl
{
	bool useHC = false;
	LZ4_streamHC_t* hcStream = nullptr;
	LZ4_stream_t* normalStream = nullptr;
	std::vector<char> tmpMemory;
};

LZ4Compressor::LZ4Compressor(int level)
	: impl_(new LZ4Compressor_Impl)
{
	if (level >= MIN_COMPRESSION && level <= MAX_COMPRESSION)
	{
		impl_->useHC = true;
		impl_->hcStream = LZ4_createStreamHC();
		LZ4_resetStreamHC_fast(impl_->hcStream, level);
	} else if (level == FAST_COMPRESSION)
	{
		impl_->useHC = false;
		impl_->normalStream = LZ4_createStream();
	}
	else
	{
		throw std::runtime_error("unknown compression level " + std::to_string(level));
	}
	impl_->tmpMemory.resize(LZ4_compressBound(MAX_CHUNK_SIZE));
}

LZ4Compressor::~LZ4Compressor()
{
	if (impl_->hcStream)
		LZ4_freeStreamHC(impl_->hcStream);
	if (impl_->normalStream)
		LZ4_freeStream(impl_->normalStream);
	delete impl_;
}

void LZ4Compressor::compress(std::ostream& dst, const char* src, int srcSize)
{
	if (srcSize > MAX_CHUNK_SIZE)
		throw std::runtime_error("src size exceeds MAX_CHUNK_SIZE");
	if (srcSize <= 0)
		throw std::runtime_error("src size must be > 0");
	int bytesWritten = 0;
	int maxDstSize = static_cast<int>(impl_->tmpMemory.size());
	if (impl_->useHC)
	{
		bytesWritten = LZ4_compress_HC_continue(impl_->hcStream, 
			src, impl_->tmpMemory.data(), srcSize, maxDstSize);
	}
	else
	{
		bytesWritten = LZ4_compress_fast_continue(impl_->normalStream,
			src, impl_->tmpMemory.data(), srcSize, maxDstSize, 1);
	}
	if (bytesWritten == 0)
		throw std::runtime_error("Error, no bytes compressed");
	dst.write(reinterpret_cast<const char*>(&bytesWritten), sizeof(int));
	dst.write(impl_->tmpMemory.data(), bytesWritten);
}

struct LZ4Decompressor_Impl
{
	LZ4_streamDecode_t* stream = nullptr;
	std::vector<char> tmpMemory;
};


LZ4Decompressor::LZ4Decompressor()
	: impl_(new LZ4Decompressor_Impl)
{
	impl_->stream = LZ4_createStreamDecode();
	impl_->tmpMemory.resize(LZ4_compressBound(LZ4Compressor::MAX_CHUNK_SIZE));
}

LZ4Decompressor::~LZ4Decompressor()
{
	if (impl_->stream)
		LZ4_freeStreamDecode(impl_->stream);
	delete impl_;
}

int LZ4Decompressor::decompress(char* dst, int dstCapacity, std::istream& src)
{
	int srcBytes;
	src.read(reinterpret_cast<char*>(&srcBytes), sizeof(int));
	src.read(impl_->tmpMemory.data(), srcBytes);
	int bytesDecompressed = LZ4_decompress_safe_continue(
		impl_->stream, impl_->tmpMemory.data(), dst, srcBytes, dstCapacity);
	return bytesDecompressed;
}

