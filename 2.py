import random
import time
import math
import sys
import bisect 

try:
    sys.setrecursionlimit(3000)
except Exception as e:
    print(f"Warning: Could not set recursion depth limit. {e}")


def generate_sorted_data(size):
  return list(range(size))

def generate_reverse_sorted_data(size):
  return list(range(size, 0, -1))

def generate_random_data(size):
  data = list(range(size))
  random.shuffle(data)
  return data

def generate_partially_sorted_data(size, sorted_percentage=0.9):
  if size == 0: return []
  sorted_part_size = int(size * sorted_percentage)
  random_part_size = size - sorted_part_size
  data = list(range(sorted_part_size))
  random_elements = random.sample(range(sorted_part_size, size * 2), random_part_size)
  data.extend(random_elements)
  random.shuffle(data)
  return data

def measure_execution_time(func, data):
  if not data: return 0.0
  start_time = time.perf_counter()
  try:
      sorted_data = func(data.copy())
  except RecursionError:
      print(f"\nRecursion depth limit reached for {func.__name__}. Increase limit or use iterative version if possible.")
      return float('inf')
  except Exception as e:
       print(f"\nError during execution of {func.__name__}: {e}")
       return float('inf') 
  end_time = time.perf_counter()
  return end_time - start_time


def swap(alist, i, j):
    """리스트의 두 원소 위치를 바꿉니다."""
    alist[i], alist[j] = alist[j], alist[i]


def max_heapify(alist, index, start, end):
    """주어진 인덱스를 루트로 하는 서브트리가 최대 힙 속성을 만족하도록 수정합니다."""
    def left(i): return 2*i + 1
    def right(i): return 2*i + 2
    size = end - start
    l = left(index)
    r = right(index)
    largest_rel_idx = index
    if (l < size and alist[start + l] > alist[start + largest_rel_idx]):
        largest_rel_idx = l
    if (r < size and alist[start + r] > alist[start + largest_rel_idx]):
        largest_rel_idx = r
    if largest_rel_idx != index:
        swap(alist, start + largest_rel_idx, start + index)
        max_heapify(alist, largest_rel_idx, start, end)

def build_max_heap(alist, start, end):
    length = end - start
    start_index = length // 2 - 1
    while start_index >= 0:
        max_heapify(alist, start_index, start, end)
        start_index = start_index - 1

def heapsort(alist, start, end):
    if end - start <= 1:
      return
    build_max_heap(alist, start, end)
    for i in range(end - 1, start, -1):
        swap(alist, start, i)
        max_heapify(alist, 0, start, i)


def library_sort_simplified(data):
    n = len(data)
    if n == 0: return []
    epsilon = 0.2; lib_size = math.ceil(n * (1 + epsilon)); library = [None] * lib_size
    elements_in_library = 0; INF_lib = float('inf') 
    for x in data:
        if elements_in_library >= lib_size * 0.9: 
            current_elements = sorted([item for item in library if item is not None])
            library = [None] * lib_size; step = lib_size / (elements_in_library + 1) if elements_in_library > 0 else 1
            idx = 0
            for i, elem in enumerate(current_elements):
                 target_idx = math.floor((i + 1) * step) -1
                 while target_idx < lib_size and library[target_idx] is not None: target_idx += 1
                 if target_idx < lib_size: library[target_idx] = elem
                 else: break
            elements_in_library = len([item for item in library if item is not None])

        temp_list = [item for item in library if item is not None] 
        pos = bisect.bisect_left(temp_list, x)
        actual_pos = 0; count = 0; found_pos = -1
        while actual_pos < lib_size:
             is_occupied = library[actual_pos] is not None
             if is_occupied:
                 if count == pos: found_pos = actual_pos; break
                 count += 1
             elif count == pos: found_pos = actual_pos; break
             actual_pos +=1
        if found_pos == -1: found_pos = actual_pos

        insert_pos = found_pos 
        while insert_pos < lib_size and library[insert_pos] is not None: insert_pos += 1
        if insert_pos < lib_size:
             shift_pos = insert_pos
             while shift_pos > found_pos: library[shift_pos] = library[shift_pos - 1]; shift_pos -= 1
             library[found_pos] = x; elements_in_library += 1
        else: 
             current_elements = [item for item in library if item is not None] + [x]
             elements_in_library = len(current_elements); library = [None] * lib_size
             step = lib_size / (elements_in_library + 1) if elements_in_library > 0 else 1
             current_elements.sort()
             for i, elem in enumerate(current_elements):
                  target_idx = math.floor((i + 1) * step) -1
                  while target_idx < lib_size and library[target_idx] is not None: target_idx += 1
                  if target_idx < lib_size: library[target_idx] = elem
                  else: break
    return [item for item in library if item is not None]

MIN_MERGE_TIM = 32
def _insertion_sort_tim(data, left, right):
    for i in range(left + 1, right + 1): key = data[i]; j = i - 1
    while j >= left and data[j] > key: data[j + 1] = data[j]; j -= 1
    data[j + 1] = key
def _merge_tim(data, l, m, r):
    len1, len2 = m - l + 1, r - m
    if len1<=0 or len2<=0: return
    left=data[l:l+len1]; right=data[m+1:m+1+len2]; i,j,k=0,0,l
    while i<len1 and j<len2:
      if left[i]<=right[j]: data[k]=left[i]; i+=1
      else: data[k]=right[j]; j+=1
      k+=1
    while i<len1: data[k]=left[i]; k+=1; i+=1
    while j<len2: data[k]=right[j]; k+=1; j+=1
def tim_sort_from_scratch(data):
    n = len(data)
    if n<2: return data
    minrun=MIN_MERGE_TIM; i=0
    while i < n:
        run_start = i; i += 1
        if i == n:
             if run_start == i-1: _insertion_sort_tim(data, run_start, min(run_start + minrun -1, n-1))
             break
        if data[i] < data[i-1]:
            while i < n and data[i] < data[i-1]: i += 1
            data[run_start:i] = data[run_start:i][::-1]
        else:
            while i < n and data[i] >= data[i-1]: i += 1
        run_end = i - 1
        if run_end - run_start + 1 < minrun:
            current_end = min(run_start + minrun - 1, n - 1)
            _insertion_sort_tim(data, run_start, current_end); i = current_end + 1
    size = minrun
    while size < n:
        for start in range(0, n, 2 * size):
            mid = min(n - 1, start + size - 1); end = min(n - 1, start + 2 * size - 1)
            if mid < end: _merge_tim(data, start, mid, end)
        size *= 2
    return data
def tim_sort_builtin(data): data.sort(); return data


def cocktail_shaker_sort(data):
    n = len(data); swapped = True; start = 0; end = n - 1
    while swapped:
        swapped = False
        for i in range(start, end):
            if data[i] > data[i + 1]: swap(data, i, i + 1); swapped = True
        if not swapped: break
        swapped = False; end -= 1
        for i in range(end - 1, start - 1, -1):
            if data[i] > data[i + 1]: swap(data, i, i + 1); swapped = True
        start += 1
    return data

def comb_sort(data):
    n = len(data); gap = n; shrink = 1.3; swapped = True
    while gap > 1 or swapped:
        gap = int(gap / shrink)
        if gap < 1: gap = 1
        swapped = False; i = 0
        while i + gap < n:
            if data[i] > data[i + gap]: swap(data, i, i + gap); swapped = True
            i += 1
    return data


import math # float('inf') 사용

# --- 전역 변수 (Python에서는 권장되지 않음) ---
n = 0
a = [] # 실제 크기는 tournament_sort에서 결정
tmp = [] # 실제 크기는 tournament_sort에서 결정
INF = float('inf') # 무한대 값 정의

# --- 헬퍼 함수 (전역 변수 사용) ---
def winner(pos1, pos2):
    """
    두 tmp 인덱스(pos1, pos2) 아래의 승자(최소값) 리프 노드의 '인덱스'를 반환.
    """
    global n, tmp # 전역 변수 사용 명시

    # 각 노드(pos1, pos2) 아래의 최종 승자(리프 노드) 인덱스를 찾음
    # 리프 인덱스는 항상 정수여야 함
    # 인덱스 유효성 체크 추가
    leaf_idx1 = -1
    if pos1 >= n:
        leaf_idx1 = pos1
    elif 0 <= pos1 < len(tmp):
        leaf_idx1 = int(tmp[pos1]) # 내부 노드 값은 리프 인덱스
    else: # pos1이 유효하지 않은 내부 노드 인덱스
         pass # leaf_idx1은 -1 유지

    leaf_idx2 = -1
    if pos2 >= n:
        leaf_idx2 = pos2
    elif 0 <= pos2 < len(tmp):
        leaf_idx2 = int(tmp[pos2])
    else: # pos2가 유효하지 않은 내부 노드 인덱스
         pass # leaf_idx2는 -1 유지

    # 리프 노드의 실제 값을 비교 (IndexError 방지)
    val1 = tmp[leaf_idx1] if (n <= leaf_idx1 < len(tmp)) else INF
    val2 = tmp[leaf_idx2] if (n <= leaf_idx2 < len(tmp)) else INF

    # 승자 결정 (최소값), 유효한 리프 인덱스 반환
    if val1 <= val2:
        # val1이 더 작거나 같으면 leaf_idx1 반환 (단, 유효한 인덱스여야 함)
        return leaf_idx1 if leaf_idx1 != -1 else leaf_idx2
    else:
        # val2가 더 작으면 leaf_idx2 반환 (단, 유효한 인덱스여야 함)
        return leaf_idx2 if leaf_idx2 != -1 else leaf_idx1

def create_tree():
    """
    전역 변수 n, a, tmp를 사용하여 초기 토너먼트 트리를 만들고,
    첫 번째 최소값과 그 값의 리프 인덱스를 반환.
    오류 시 (INF, -1) 반환.
    """
    global n, a, tmp, INF # 전역 변수 사용 명시

    # 리프 노드에 데이터 복사 (인덱스 n ~ 2n-1)
    for i in range(n):
        if n + i < len(tmp):
            tmp[n + i] = a[i]
        else:
            print(f"Error: Index out of bounds tmp[{n+i}] (n={n}).")
            return INF, -1

    # 내부 노드 채우기 (인덱스 n-1 부터 0 까지, bottom-up)
    # 루프 범위를 n-1 부터 0 까지로 수정해야 함
    for k in range(n - 1, -1, -1):
         child1_idx = 2 * k + 1
         child2_idx = 2 * k + 2

         # 자식 노드 존재 여부 확인
         has_child1 = child1_idx < 2 * n
         has_child2 = child2_idx < 2 * n

         if has_child1 and has_child2: # 두 자식 모두 존재
             tmp[k] = winner(child1_idx, child2_idx)
         elif has_child1: # 왼쪽 자식만 존재 (n이 홀수일 때 마지막 부모)
             # 왼쪽 자식 아래의 승자 인덱스를 그대로 가져옴
             tmp[k] = child1_idx if child1_idx >= n else tmp[child1_idx]
         # else: 자식이 없는 경우 (k >= n), 이 루프에서 처리 안함

    # 첫 번째 승자 정보 반환
    if n > 0:
        # 루트(tmp[0])에 저장된 최종 승자(리프) 인덱스
        first_winner_leaf_idx = int(tmp[0])
        # 인덱스 유효성 확인 후 값 반환
        if n <= first_winner_leaf_idx < len(tmp):
            first_value = tmp[first_winner_leaf_idx]
            return first_value, first_winner_leaf_idx
        else:
             print(f"Error: Invalid winner index {first_winner_leaf_idx} at tmp[0].")
             return INF, -1
    else:
        return INF, -1 # 빈 입력

def recreate(last_winner_leaf_idx):
    """
    이전 승자의 리프 인덱스를 받아 트리를 업데이트하고,
    다음 최소값과 그 리프 인덱스를 반환.
    오류 시 (INF, -1) 반환.
    (주의: 이 함수 호출 전에 tmp[last_winner_leaf_idx] = INF 처리가 필요함)
    """
    global n, tmp, INF # 전역 변수 사용

    # 유효하지 않은 인덱스 입력 방지
    if not (n <= last_winner_leaf_idx < 2 * n):
         print(f"Error: Invalid index {last_winner_leaf_idx} passed to recreate.")
         return INF, -1

    current_idx = last_winner_leaf_idx
    # 리프에서 루트(0)까지 올라가면서 부모 노드 업데이트
    while current_idx > 0:
        parent_idx = (current_idx - 1) // 2

        # 형제 노드 인덱스 계산
        if current_idx % 2 == 1: # 현재가 왼쪽 자식
            sibling_idx = current_idx + 1
        else: # 현재가 오른쪽 자식
            sibling_idx = current_idx - 1

        # 형제 노드 존재 여부 확인
        has_sibling = (0 <= sibling_idx < 2 * n)

        # 부모 노드의 승자 다시 결정
        if has_sibling:
            tmp[parent_idx] = winner(current_idx, sibling_idx)
        else:
            # 형제가 없으면, 부모의 승자는 현재 노드 아래의 승자 인덱스여야 함.
            # 현재 노드 아래는 이미 INF 처리되었으므로, 부모도 이 인덱스를 가리키게 됨.
             tmp[parent_idx] = current_idx if current_idx >= n else tmp[current_idx]

        current_idx = parent_idx # 위로 이동

    # 새로운 승자 정보 반환 (루트 tmp[0] 확인)
    next_winner_leaf_idx = int(tmp[0])
    if n <= next_winner_leaf_idx < len(tmp):
        next_value = tmp[next_winner_leaf_idx]
        return next_value, next_winner_leaf_idx
    else:
        # 모든 요소가 INF 처리되었거나 오류
        return INF, -1

def tournament_sort(data):
    """
    사용자 의사코드 로직 기반 토너먼트 정렬 실행 함수.
    전역 변수 n, a, tmp 를 사용 및 수정.
    정렬된 **새 리스트**를 반환 (원본 data는 변경 안함).
    """
    global n, a, tmp, INF # 전역 변수 사용 선언

    n = len(data)
    if n <= 1:
        return data[:] # 원본의 복사본 반환

    # 전역 변수 초기화
    a = data[:] # 입력 데이터 복사 (필요 시)
    tmp = [0.0] * (2 * n) # INF 저장을 위해 float으로 초기화

    # 1. 초기 트리 생성 및 첫 번째 최소값 얻기
    current_value, current_winner_idx = create_tree()

    # 오류 체크
    if current_value == INF or current_winner_idx == -1 :
        print("Error during initial tree creation.")
        return data[:] # 오류 시 원본 복사본 반환

    # 결과 저장용 리스트
    sorted_list = [0] * n

    # 2. 반복적으로 최소값 추출 및 트리 재구성
    for i in range(n):
        # 현재 찾은 최소값을 결과 리스트에 저장
        sorted_list[i] = current_value

        # 방금 찾은 최소값의 리프 노드를 INF로 표시 (다음 recreate 준비)
        if n <= current_winner_idx < len(tmp):
            tmp[current_winner_idx] = INF
        else:
             print(f"Error: Invalid index {current_winner_idx} for winner at step {i}.")
             # 오류 처리: 현재까지 정렬된 부분만 반환하거나 빈 리스트 반환
             return sorted_list[:i]

        # 마지막 요소가 아니면 다음 최소값 찾기
        if i < n - 1:
            current_value, current_winner_idx = recreate(current_winner_idx)
            # recreate 오류 체크
            if current_value == INF or current_winner_idx == -1 :
                print(f"Error: Could not find next winner at step {i+1}.")
                # 오류 처리
                return sorted_list[:i+1] # 현재까지 찾은 값만 반환

    # 최종 정렬된 리스트 반환
    return sorted_list


def _insertion_sort_range(alist, start, end):
    for i in range(start + 1, end):
        key = alist[i]
        j = i - 1
        while j >= start and key < alist[j]:
            alist[j + 1] = alist[j]
            j -= 1
        alist[j + 1] = key


def max_heapify(alist, index, start, end):
    def left(i): return 2*i + 1
    def right(i): return 2*i + 2
    size = end - start
    l = left(index)
    r = right(index)
    largest_rel_idx = index
    actual_idx = start + index
    largest_actual_idx = start + largest_rel_idx

    if l < size and alist[start + l] > alist[largest_actual_idx]:
        largest_rel_idx = l
        largest_actual_idx = start + l 
    if r < size and alist[start + r] > alist[largest_actual_idx]:
        largest_rel_idx = r
        largest_actual_idx = start + r 

    if largest_rel_idx != index: 
        swap(alist, largest_actual_idx, actual_idx)
        max_heapify(alist, largest_rel_idx, start, end) 

def build_max_heap(alist, start, end):
    length = end - start
    start_index = length // 2 - 1 
    while start_index >= 0:
        max_heapify(alist, start_index, start, end)
        start_index = start_index - 1

def _heap_sort_range(alist, start, end):
    if end - start <= 1:
        return
    build_max_heap(alist, start, end)
    for i in range(end - 1, start, -1):
        swap(alist, start, i) 
        max_heapify(alist, 0, start, i) 

def _median_of_three(alist, low, high_exclusive):
    mid = low + (high_exclusive - 1 - low) // 2
    if alist[mid] < alist[low]:
        swap(alist, mid, low)
    if alist[high_exclusive - 1] < alist[low]:
        swap(alist, high_exclusive - 1, low)
    if alist[mid] < alist[high_exclusive - 1]:
        swap(alist, mid, high_exclusive - 1)
    swap(alist, low, high_exclusive - 1)
    return low 

def _partition_lomuto(alist, low, high_exclusive):

    if high_exclusive - low >= 3:
        pivot_index = _median_of_three(alist, low, high_exclusive)
    else:
        # 크기가 작으면 그냥 첫번째 요소를 피벗으로 사용
        pivot_index = low # 이미 첫번째 요소임

    pivot = alist[pivot_index] # 피벗 값
    # 실제 Lomuto 파티션은 피벗을 맨 끝으로 보내는 경우가 많으나,
    # 여기서는 median_of_three가 맨 앞에 뒀으므로 첫 요소를 기준으로 파티션
    i = low # 피벗보다 작거나 같은 요소들의 마지막 위치 + 1
    for j in range(low + 1, high_exclusive):
        if alist[j] < pivot:
            i += 1
            swap(alist, i, j)
    # 마지막으로 피벗(원래 alist[low])을 i 위치와 교환하여 최종 위치 확정
    swap(alist, low, i)
    return i # 피벗의 최종 위치 반환

def _introsort_recursive(alist, start, end, maxdepth):
    """표준 Introsort 로직을 따르는 재귀 함수"""
    size = end - start

    # 의사코드 규칙 1: 크기가 16 미만이면 삽입 정렬
    if size < 16:
        _insertion_sort_range(alist, start, end)
        return

    # 의사코드 규칙 2: 재귀 깊이 한계 도달 시 힙 정렬
    if maxdepth == 0:
        _heap_sort_range(alist, start, end)
        return

    # 의사코드 규칙 3: 퀵 정렬 파티션 및 재귀 호출
    # Lomuto 파티션 사용 (피벗의 최종 위치 p 반환)
    p = _partition_lomuto(alist, start, end)
    # 재귀 호출 (p 위치는 정렬 완료)
    _introsort_recursive(alist, start, p, maxdepth - 1) # 왼쪽 부분: start ~ p-1
    _introsort_recursive(alist, p + 1, end, maxdepth - 1) # 오른쪽 부분: p+1 ~ end-1

def introsort_standard(alist):
    """표준 Introsort 알고리즘 구현 (의사코드 기반)"""
    n = len(alist)
    if n <= 1:
        return alist # 이미 정렬됨

    # 최대 깊이 계산: 2 * floor(log2(n))
    maxdepth = (n.bit_length() - 1) * 2 if n > 0 else 0
    _introsort_recursive(alist, 0, n, maxdepth)
    return alist # 정렬된 리스트 반환

# --- 실험 설정 ---
# data_sizes = [100, 1000, 5000] # 간소화 버전 고려 시 추천 크기
data_sizes = [1000, 10000] # 조금 더 크게 테스트
num_runs = 3

# 테스트할 정렬 알고리즘 딕셔너리
sorting_algorithms = {
    "Library Sort (Simp.)": library_sort_simplified,
    "Tim Sort (Scratch Simp.)": tim_sort_from_scratch,
    "Tim Sort (Built-in)": tim_sort_builtin,
    "Cocktail Shaker Sort": cocktail_shaker_sort,
    "Comb Sort": comb_sort,
    "Tournament Sort (Heap)": tournament_sort,
    "Introsort (User Rules)": introsort_standard,
}

# 잠재적으로 느릴 수 있는 알고리즘 목록 (필요 시 건너뛰기 용)
potentially_slow_algorithms = [
    "Library Sort (Simp.)",
    "Tim Sort (Scratch Simp.)",
    "Cocktail Shaker Sort", # O(n^2)
    # "Introsort (User Rules)" # 최종 삽입 정렬 때문에 큰 데이터에서 느릴 수 있음
]

# --- 데이터 생성 ---
input_data = {}
print("Generating test data...")
for size in data_sizes:
  print(f"  Generating size {size}...")
  input_data[f"sorted_{size}"] = generate_sorted_data(size)
  input_data[f"reverse_sorted_{size}"] = generate_reverse_sorted_data(size)
  input_data[f"random_{size}"] = generate_random_data(size)
  input_data[f"partially_sorted_{size}"] = generate_partially_sorted_data(size)
print("Data generation complete.")
print("-" * 50)

# --- 실험 실행 및 결과 기록 ---
results = {}
for data_name, data_list in input_data.items():
  current_data_size = len(data_list)
  print(f"Running tests on: {data_name} (Size: {current_data_size})")
  results[data_name] = {}

  for algo_name, algo_func in sorting_algorithms.items():
      # 큰 데이터셋에서 잠재적으로 느린 알고리즘 건너뛰기 (옵션)
      skip_threshold = 500000 # 예: 5000개 초과 시 건너뛸 임계값
      if algo_name in potentially_slow_algorithms and current_data_size > skip_threshold:
          print(f"  Skipping {algo_name} for dataset size > {skip_threshold}")
          results[data_name][algo_name] = float('inf')
          continue

      execution_times = []
      print(f"  Running {algo_name}...", end='')
      skip_remaining_runs = False
      for i in range(num_runs):
          execution_time = measure_execution_time(algo_func, data_list)
          if execution_time == float('inf'):
              execution_times = [float('inf')]
              skip_remaining_runs = True # 오류 발생 시 중단
              break
          execution_times.append(execution_time)
          # 시간 제한 로직 (옵션)
          # if sum(execution_times) > 60: # 예: 60초
          #    print(" -> Timeout (>60s)", end='')
          #    execution_times = [float('inf')]
          #    skip_remaining_runs = True
          #    break

      if skip_remaining_runs:
          mean_time = float('inf')
          print(f" -> FAILED or TIMEOUT")
      elif execution_times:
          mean_time = sum(execution_times) / len(execution_times)
          print(f" -> {mean_time:.6f} seconds (Avg of {len(execution_times)} runs)")
      else:
          mean_time = 0.0
          print(" -> 0.000000 seconds (No runs executed)")

      results[data_name][algo_name] = mean_time
  print("-" * 50)

# --- 최종 결과 출력 ---
print("\n--- Mean Execution Time Results (seconds) ---")
algo_names = list(sorting_algorithms.keys())
col_width = 26 # 컬럼 너비

header = "Data Set".ljust(col_width) + "".join(name.ljust(col_width) for name in algo_names)
print(header)
print("-" * len(header))

for data_name, algo_results in results.items():
    row = data_name.ljust(col_width)
    for algo_name in algo_names:
        mean_time = algo_results.get(algo_name, float('inf'))
        if mean_time == float('inf'):
            time_str = "Inf/Skip/Fail".ljust(col_width)
        else:
            time_str = f"{mean_time:.6f}".ljust(col_width)
        row += time_str
    print(row)

print("-" * len(header))

# 프로그램 실행을 위한 메인 가드 (스크립트로 실행 시 테스트 시작)
# if __name__ == '__main__':
#    pass # 위에서 이미 실험 코드가 전역 범위에서 실행됨