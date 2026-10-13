# Fails unless every opcode in OPCODES appears in the SASS of BINARY.
# Usage: cmake -DBINARY=<exe> -DCUOBJDUMP=<cuobjdump> -DOPCODES="A;B" -P check_sass.cmake
execute_process(COMMAND ${CUOBJDUMP} -sass ${BINARY} OUTPUT_VARIABLE sass RESULT_VARIABLE rc)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "cuobjdump failed on ${BINARY}")
endif()
foreach(op IN LISTS OPCODES)
  string(FIND "${sass}" "${op}" pos)
  if(pos EQUAL -1)
    message(FATAL_ERROR "${op} not found in the SASS of ${BINARY}")
  endif()
  message(STATUS "found ${op}")
endforeach()
