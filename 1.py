import random
import time

def generate_sorted_data(size):
  """오름차순으로 정렬된 데이터를 생성합니다."""
  return list(range(size))

def generate_reverse_sorted_data(size):
  """내림차순으로 정렬된 데이터를 생성합니다."""
  return list(range(size, 0, -1))

def generate_random_data(size):
  """무작위로 섞인 데이터를 생성합니다."""
  data = list(range(size))
  random.shuffle(data)
  return data

def generate_partially_sorted_data(size, sorted_percentage=0.9):
  """부분적으로 정렬된 데이터를 생성합니다."""
  sorted_part_size = int(size * sorted_percentage)
  random_part_size = size - sorted_part_size
  data = list(range(sorted_part_size)) + random.sample(range(size * 2), random_part_size)
  random.shuffle(data)
  return data

def measure_execution_time(func, data):
  """주어진 함수(정렬 알고리즘)의 실행 시간을 측정합니다."""
  start_time = time.time()
  func(data.copy()) # 원본 데이터 보존을 위해 복사본 사용
  end_time = time.time()
  return end_time - start_time

# --- 정렬 알고리즘 구현 ---

def bubble_sort(data):
  """버블 정렬"""
  n = len(data)
  for i in range(n - 1):
    for j in range(n - i - 1):
      if data[j] > data[j + 1]:
        data[j], data[j + 1] = data[j + 1], data[j]
  return data

def insertion_sort(data):
  """삽입 정렬"""
  n = len(data)
  for i in range(1, n):
    key = data[i]
    j = i - 1
    while j >= 0 and key < data[j]:
      data[j + 1] = data[j]
      j -= 1
    data[j + 1] = key
  return data

def selection_sort(data):
  """선택 정렬"""
  n = len(data)
  for i in range(n):
    min_index = i
    for j in range(i + 1, n):
      if data[j] < data[min_index]:
        min_index = j
    data[i], data[min_index] = data[min_index], data[i]
  return data

def merge_sort(data):
  """병합 정렬"""
  if len(data) <= 1:
    return data
  mid = len(data) // 2
  left = data[:mid]
  right = data[mid:]
  return merge(merge_sort(left), merge_sort(right))

def merge(left, right):
  """병합 정렬의 서브루틴"""
  merged = []
  left_index = 0
  right_index = 0
  while left_index < len(left) and right_index < len(right):
    if left[left_index] <= right[right_index]:
      merged.append(left[left_index])
      left_index += 1
    else:
      merged.append(right[right_index])
      right_index += 1
  merged.extend(left[left_index:])
  merged.extend(right[right_index:])
  return merged

def heapify(data, n, i):
  """힙 정렬의 서브루틴: 힙 속성 유지"""
  largest = i
  left = 2 * i + 1
  right = 2 * i + 2

  if left < n and data[left] > data[largest]:
    largest = left

  if right < n and data[right] > data[largest]:
    largest = right

  if largest != i:
    data[i], data[largest] = data[largest], data[i]
    heapify(data, n, largest)

def heap_sort(data):
  """힙 정렬"""
  n = len(data)
  # 초기 힙 구성 (바텀업 방식)
  for i in range(n // 2 - 1, -1, -1):
    heapify(data, n, i)
  # 정렬: 루트 노드를 마지막 요소와 교환하고 힙 크기를 줄여가며 힙 재구성
  for i in range(n - 1, 0, -1):
    data[i], data[0] = data[0], data[i]
    heapify(data, i, 0)
  return data

def quick_sort(data):
  """퀵 정렬"""
  if len(data) <= 1:
    return data
  pivot = data[len(data) // 2]
  left = [x for x in data if x < pivot]
  middle = [x for x in data if x == pivot]
  right = [x for x in data if x > pivot]
  return quick_sort(left) + middle + quick_sort(right)

# --- 실험 설정 ---

data_sizes = [1000, 10000, 100000, 1000000]
num_runs = 10

sorting_algorithms = {
    "Bubble Sort": bubble_sort,
    "Insertion Sort": insertion_sort,
    "Selection Sort": selection_sort,
    "Merge Sort": merge_sort,
    "Heap Sort": heap_sort,
    "Quick Sort": quick_sort,
}

input_data = {}
for size in data_sizes:
  input_data[f"sorted_{size}"] = generate_sorted_data(size)
  input_data[f"reverse_sorted_{size}"] = generate_reverse_sorted_data(size)
  input_data[f"random_{size}"] = generate_random_data(size)
  input_data[f"partially_sorted_{size}"] = generate_partially_sorted_data(size)

# --- 실험 실행 및 결과 기록 ---

results = {}

for data_name, data_list in input_data.items():
  print(f"Running tests on: {data_name}")
  results[data_name] = {}
  for algo_name, algo_func in sorting_algorithms.items():
    execution_times = []
    for i in range(num_runs):
      print(f"  Running {algo_name} (Run {i+1})...", end='\r')
      execution_time = measure_execution_time(algo_func, data_list)
      execution_times.append(execution_time)
    mean_time = sum(execution_times) / num_runs
    results[data_name][algo_name] = mean_time
    print(f"  {algo_name}: {mean_time:.6f} seconds (Average of {num_runs} runs)")
  print("-" * 50)

# --- 최종 결과 출력 ---

print("\n--- Mean Execution Time Results ---")
for data_name, algo_results in results.items():
  print(f"Data: {data_name}")
  for algo_name, mean_time in algo_results.items():
    print(f"  {algo_name}: {mean_time:.6f} seconds")
  print("-" * 50)