#include <cstring>
#define errormessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)

#ifdef DEBUG_SYCL
#define debugmessage(x, ...) do { fprintf(stderr, "%s:%d " x, __FILE__, __LINE__, __VA_ARGS__ ); } while (0)
#else
#define debugmessage(x, ...)
#endif

#include "syclCommon.hpp"

#ifdef WITH_ONEAPI_ONECCL

#include <oneapi/ccl.hpp>
#include <mpi.h>

using namespace sycl_be;

extern "C" {

  int onecclGroupStartFromC() {
    ccl::group_start();
    return 1;
  }

  int onecclGroupEndFromC() {
    ccl::group_end();
    return 1;
  }

  int onecclInitFromC() {
    ccl::init();
    return 1;
  }

  /**
   * Create a main Key-Value Store to create a oneCCL communicator.
   * Only call from a single "root" rank of your future communicator!
   */
  int onecclGetUniqueIdFromC(char *kvsAddress) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ccl::shared_ptr_class<ccl::kvs> kvs;
    kvs = ccl::create_main_kvs();
    ccl::kvs::address_type tmpAddr = kvs->get_address();
    // std::cerr << "[%%%] onecclGetUniqueIdFromC called from rank " << rank << ", addr: "  << tmpAddr.data() << std::endl;
    std::memcpy(kvsAddress, tmpAddr.data(), tmpAddr.max_size());
    std::string kvsAddrStr(kvsAddress);
    // So that we can retrieve the KVS later and it doesn't get deconstructed. The Fortran Layer only knows about the address string. 
    SyclState::defaultState().registerKvs(kvsAddrStr, kvs);
    return 1;
  }


  int onecclCommInitRankFromC(ccl::communicator **onecclComm, int nRanks, char *kvsAddress, int myRank) {
    SyclState &ss = SyclState::defaultState();

    std::string kvsAddrStr(kvsAddress);
    //std::cerr << "[%%%] onecclCommInitRankFromC called from rank " << myRank << " for nRanks " << nRanks <<", addr: "  << kvsAddrStr << std::endl;
    std::optional<cclKvsHandle> kvsOpt = ss.retrieveKvs(kvsAddrStr);
    cclKvsHandle kvs;
    if (!kvsOpt.has_value()) {
      ccl::kvs::address_type kvsAddr;
      std::memcpy(kvsAddr.data(), kvsAddress, kvsAddr.max_size());
      kvs = ccl::create_kvs(kvsAddr);
      ss.registerKvs(kvsAddrStr, kvs);
    } else {
      kvs = kvsOpt.value();
    }
    // oneCCL doesn't return an opaque pointer to the communicator, but an interface object instead.
    *onecclComm = ss.getDefaultDeviceHandle().initCclCommunicator(nRanks, myRank, kvs);
    std::cout << "[%%%] CCL Communicator created for rank " << myRank << " of " << nRanks
              << " with address: " << std::hex << reinterpret_cast<std::intptr_t>(*onecclComm) << std::dec << "(" << (*onecclComm)->rank() << "/" << (*onecclComm)->size() << ")" << std::endl;

    return 1;
  }

  int onecclCommDestroyFromC(ccl::communicator *onecclComm, QueueData *) {
    delete onecclComm;
    return 1;
  }

  int onecclStreamSynchronizeFromC(QueueData *qd) {
    sycl::queue q = getQueueOrDefault(qd);
    q.wait();
    return 1;
  }

  int onecclRedOpSumFromC() {
    return static_cast<int>(ccl::reduction::sum);
  }

  int onecclRedOpProdFromC() {
    return static_cast<int>(ccl::reduction::prod);
  }

  int onecclRedOpMinFromC() {
    return static_cast<int>(ccl::reduction::min);
  }

  int onecclRedOpMaxFromC() {
    return static_cast<int>(ccl::reduction::max);
  }

  int onecclRedOpAvgFromC(void) {
    return static_cast<int>(ccl::reduction::avg);
  }

  int onecclDataTypeOnecclIntFromC(void) {
    // According to the NVIDIA Docs, this is supposed to be an int32.
    // There is no direct match in oneCCL, thus return the int32 enum value.
    return static_cast<int>(ccl::datatype::int32);
  }

  int onecclDataTypeOnecclInt32FromC(void) {
    return static_cast<int>(ccl::datatype::int32);
  }

  int onecclDataTypeOnecclInt64FromC(void) {
    return static_cast<int>(ccl::datatype::int64);
  }

  int onecclDataTypeOnecclFloat32FromC(void) {
    return static_cast<int>(ccl::datatype::float32);
  }

  int onecclDataTypeOnecclFloatFromC(void) {
    // Same as above, no direct match, return value according to specification in the NVIDIA docs.
    return static_cast<int>(ccl::datatype::float32);
  }

  int onecclDataTypeOnecclFloat64FromC(void) {
    return static_cast<int>(ccl::datatype::float64);
  }

  int onecclDataTypeOnecclDoubleFromC(void) {
    // Same as above, no direct match, return value according to specification in the NVIDIA docs.
    return static_cast<int>(ccl::datatype::float64);
  }

  size_t onecclSizeForDatatypeFromC(ccl::datatype onecclDatatype) {
    switch (onecclDatatype) {
      case ccl::datatype::int32:
        return sizeof(int32_t);
      case ccl::datatype::int64:
        return sizeof(int64_t);
      case ccl::datatype::float32:
        return sizeof(float);
      case ccl::datatype::float64:
        return sizeof(double);
      default:
      errormessage("%s\n", "Error in onecclSizeForDatatype: Unknown datatype.");
        return 0;
    }
  }

  int onecclAllReduceFromC(const void *sendbuff, void *recvbuff, size_t count, ccl::datatype onecclDatatype, ccl::reduction onecclOp, ccl::communicator *onecclComm, QueueData *qd) {
    QueueData *qData = getQueueDataOrDefault(qd);
    if (onecclOp == ccl::reduction::custom) {
      errormessage("%s\n", "Error in onecclAllReduce: ccl::reduction::custom is not supported in ELPA.");
      return 0;
    }

    try {
      auto attributes = ccl::create_operation_attr<ccl::allreduce_attr>();
      auto &comm = *onecclComm;
      auto &stream = qData->cclStream;
      std::cout << "[%%%] onecclAllReduce called with count " << count
            << ", datatype " << static_cast<int>(onecclDatatype)
            << ", op " << static_cast<int>(onecclOp)
            << ", sendbuff: " << std::hex << reinterpret_cast<uintptr_t>(sendbuff) << std::dec
            << ", recvbuff: " << std::hex << reinterpret_cast<uintptr_t>(recvbuff) << std::dec
            << ", comm: " << comm.rank() << "/" << comm.size()
            << ", stream: " << &stream << std::endl;
      ccl::allreduce(sendbuff, recvbuff, count, onecclDatatype, onecclOp, comm, stream, attributes).wait();
    } catch (const ccl::exception &e) {
      errormessage("Error in onecclAllReduce: %s\n", e.what());
      return 0;
    }
    return 1;
  }

  int onecclReduceFromC(const void *sendbuff, void *recvbuff, size_t count, ccl::datatype onecclDatatype, ccl::reduction onecclOp, int root, ccl::communicator *onecclComm, QueueData *qd) {
    QueueData *qData = getQueueDataOrDefault(qd);
    if (onecclOp == ccl::reduction::custom) {
      errormessage("%s\n", "Error in onecclReduce: ccl::reduction::custom is not supported in ELPA. (Likely you wanted avg, which oneCCL doesn't have)");
      return 0;
    }

    try {
      auto attributes = ccl::create_operation_attr<ccl::reduce_attr>();
      ccl::reduce(sendbuff, recvbuff, count, onecclDatatype, onecclOp, root, *onecclComm, qData->cclStream, attributes).wait();
    } catch (const ccl::exception &e) {
      errormessage("Error in onecclReduce: %s\n", e.what());
      return 0;
    }
    return 1;
  }

  int onecclBroadcastFromC(void* sendbuff, void* recvbuff, size_t count, ccl::datatype onecclDatatype, int root, ccl::communicator *onecclComm, QueueData *qd) {
    try {
      QueueData *qData = getQueueDataOrDefault(qd);
      ccl::broadcast(sendbuff, recvbuff, count, onecclDatatype, root, *onecclComm, qData->cclStream).wait();
    } catch (const ccl::exception &e) {
      errormessage("Error in onecclBroadcast: %s\n", e.what());
      return 0;
    }
    return 1;
  }

  int onecclSendFromC(void* sendbuff, size_t count, ccl::datatype onecclDatatype, int peer, ccl::communicator *onecclComm, QueueData *qd) {
    try {
      QueueData *qData = getQueueDataOrDefault(qd);
      ccl::stream &stream = qData->cclStream;
      ccl::send(sendbuff, count, onecclDatatype, peer, *onecclComm, stream).wait();
    } catch (const ccl::exception &e) {
      errormessage("Error in onecclSend: %s\n", e.what());
      return 0;
    }
    return 1;
  }

  int onecclRecvFromC(void* recvbuff, size_t count, ccl::datatype onecclDatatype, int peer, ccl::communicator *onecclComm, QueueData *qd) {
    try {
      QueueData *qData = getQueueDataOrDefault(qd);
      ccl::recv(recvbuff, count, onecclDatatype, peer, *onecclComm, qData->cclStream).wait();
    } catch (const ccl::exception &e) {
      errormessage("Error in onecclRecv: %s\n", e.what());
      return 0;
    }
    return 1;
  }

}
#endif
