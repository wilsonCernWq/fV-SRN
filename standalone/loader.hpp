#pragma once


#include <cassert>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>

#include "lz4cpp.hpp"

#include "vidi_filemap.h"

struct GridDesc {
    std::vector<char> data;
    int dims[3];
    enum DataType { TypeUChar, TypeUShort, TypeFloat, _TypeCount_ } type;
};

namespace loader {

static const char MAGIC[] = "CVOL";
static const char MAGIC_OLD[] = "cvol";
static const int VERSION = 1;


enum Flags {
  Flag_Compressed = 1,
  // more flags to be added in the future
};

enum DataType { TypeUChar, TypeUShort, TypeFloat, _TypeCount_ };
static const int BytesPerType[_TypeCount_] = { 1, 2, 4 };

class Feature;
typedef std::shared_ptr<Feature> Feature_ptr;

class Feature
{
public:
  const std::string name_;
  const DataType type_;
  const int numChannels_;
  uint64_t sizeX;
  uint64_t sizeY;
  uint64_t sizeZ;
  uint64_t nbytes;

  std::unique_ptr<char[]> dataCpu_;

  Feature(const std::string& name, DataType type, int numChannels, uint64_t sizeX, uint64_t sizeY, uint64_t sizeZ) : name_(name), type_(type), numChannels_(numChannels)
  {
    this->sizeX = sizeX;
    this->sizeY = sizeY;
    this->sizeZ = sizeZ;

    dataCpu_.reset(new char[numChannels * sizeX * sizeY * sizeZ * BytesPerType[type]]);
    nbytes = numChannels * sizeX * sizeY * sizeZ * BytesPerType[type];
  }

  static std::shared_ptr<Feature> load(std::ifstream& s, LZ4Decompressor* compressor)
  {
    int lenName;
    std::string name;
    uint64_t sizeX, sizeY, sizeZ;
    int channels;
    int type;

    s.read(reinterpret_cast<char*>(&lenName), 4);
    name.resize(lenName);
    s.read((char*)name.data(), lenName);
    s.read(reinterpret_cast<char*>(&sizeX), 8);
    s.read(reinterpret_cast<char*>(&sizeY), 8);
    s.read(reinterpret_cast<char*>(&sizeZ), 8);
    s.read(reinterpret_cast<char*>(&channels), 4);
    s.read(reinterpret_cast<char*>(&type), 4);

    Feature_ptr f = std::make_shared<Feature>(name, static_cast<DataType>(type), channels, sizeX, sizeY, sizeZ);
    auto& data = f->dataCpu_;
    bool useCompression = compressor != nullptr;

    // body
    if (useCompression) {
      size_t dataToRead = BytesPerType[type] * sizeX * sizeY * sizeZ * channels;
      for (size_t offset = 0; offset < dataToRead;) {
        char* mem = data.get() + offset;
        const int len = std::min(static_cast<int>(dataToRead - offset), std::numeric_limits<int>::max());
        int chunkSize = compressor->decompress(mem, len, s);
        offset += chunkSize;
      }
    }
    else {
      size_t dataToRead = BytesPerType[type] * sizeX * sizeY * channels;
      for (int z = 0; z < sizeZ; ++z) {
        s.read(data.get() + z * dataToRead, dataToRead);
      }
    }

    return f;
  }
};

GridDesc
load(std::string filename)
{
  float worldSizeX_, worldSizeY_, worldSizeZ_;
  std::vector<Feature_ptr> features_;

  assert(sizeof(size_t) == 8);
  assert(sizeof(double) == 8);
  std::ifstream s(filename, std::fstream::binary);
  if (!s.is_open()) {
    throw std::runtime_error("Unable to open file " + filename);
  }

  // header
  char magic[4];
  s.read(magic, 4);

  if (memcmp(MAGIC, magic, 4) == 0) {
    // load new version
    int version;
    int numFeatures;
    int flags;
    s.read(reinterpret_cast<char*>(&version), 4);
    if (version != VERSION) {
      throw std::runtime_error("Unknown file version!");
    }
    s.read(reinterpret_cast<char*>(&worldSizeX_), 4);
    s.read(reinterpret_cast<char*>(&worldSizeY_), 4);
    s.read(reinterpret_cast<char*>(&worldSizeZ_), 4);
    s.read(reinterpret_cast<char*>(&numFeatures), 4);
    s.read(reinterpret_cast<char*>(&flags), 4);
    s.ignore(4);
    bool useCompression = (flags & Flag_Compressed) > 0;

    LZ4Decompressor d;
    LZ4Decompressor* dPtr = useCompression ? &d : nullptr;

    // load features
    features_.resize(numFeatures);
    const float progressStep = 1.0f / numFeatures;
    for (int i = 0; i < numFeatures; ++i) {
      std::cout << "Load feature " << std::to_string(i) << std::endl;
      const auto f = Feature::load(s, dPtr);
      features_[i] = f;
    }
  }
  else if (memcmp(MAGIC_OLD, magic, 4) == 0) {
    // old version, only density
    size_t sizeX, sizeY, sizeZ;
    double voxelSizeX, voxelSizeY, voxelSizeZ;
    char useCompression;
    s.read(reinterpret_cast<char*>(&sizeX), 8);
    s.read(reinterpret_cast<char*>(&sizeY), 8);
    s.read(reinterpret_cast<char*>(&sizeZ), 8);
    s.read(reinterpret_cast<char*>(&voxelSizeX), 8);
    s.read(reinterpret_cast<char*>(&voxelSizeY), 8);
    s.read(reinterpret_cast<char*>(&voxelSizeZ), 8);
    unsigned int type;
    s.read(reinterpret_cast<char*>(&type), 4);
    s.read(&useCompression, 1);
    s.ignore(7);

    // create feature and level
    Feature_ptr f = std::make_shared<Feature>("level", static_cast<DataType>(type), 1, sizeX, sizeY, sizeZ);
    features_.push_back(f);
    auto& data = f->dataCpu_;

    worldSizeX_ = voxelSizeX * sizeX;
    worldSizeY_ = voxelSizeY * sizeY;
    worldSizeZ_ = voxelSizeZ * sizeZ;

    // body
    if (useCompression) {
      LZ4Decompressor d;
      size_t dataToRead = BytesPerType[type] * sizeX * sizeY * sizeZ;
      for (size_t offset = 0; offset < dataToRead;) {
        char* mem = data.get() + offset;
        const int len = std::min(static_cast<int>(dataToRead - offset), std::numeric_limits<int>::max());
        int chunkSize = d.decompress(mem, len, s);
        offset += chunkSize;
      }
    }
    else {
      size_t dataToRead = BytesPerType[type] * sizeX * sizeY;
      for (int z = 0; z < sizeZ; ++z) {
        s.read(data.get() + z * dataToRead, dataToRead);
      }
    }
  }
  else {
    throw std::runtime_error("Illegal magic number");
  }

  std::cout << "done reading " << filename << std::endl;
  std::cout << "# of features = " << features_.size() << std::endl;
  std::cout << "worldSizeX_ " << worldSizeX_ << std::endl;
  std::cout << "worldSizeY_ " << worldSizeY_ << std::endl;
  std::cout << "worldSizeZ_ " << worldSizeZ_ << std::endl;

  GridDesc ret;
  if (features_.size() != 1) throw std::runtime_error("wrong number of variables");
  {
    ret.dims[0] = features_[0]->sizeX;
    ret.dims[1] = features_[0]->sizeY;
    ret.dims[2] = features_[0]->sizeZ;
    switch (features_[0]->type_) {
    case TypeUChar:  ret.type = GridDesc::TypeUChar;  break;
    case TypeUShort: ret.type = GridDesc::TypeUShort; break;
    case TypeFloat:  ret.type = GridDesc::TypeFloat;  break;
    default: throw std::runtime_error("wrong type");
    }

    std::cout << "numchannels " << features_[0]->numChannels_ << std::endl;
    ret.data.resize(features_[0]->nbytes);
    std::memcpy(ret.data.data(), features_[0]->dataCpu_.get(), features_[0]->nbytes);
  }

  return ret;
}


}
