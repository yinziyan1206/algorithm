import bisect
from typing import List, Set


# 815
class Solution815:
    """
        给你一个数组 routes ，表示一系列公交线路，其中每个 routes[i] 表示一条公交线路，第 i 辆公交车将会在上面循环行驶。

        例如，路线 routes[0] = [1, 5, 7] 表示第 0 辆公交车会一直按序列 1 -> 5 -> 7 -> 1 -> 5 -> 7 -> 1 -> ... 这样的车站路线行驶。
        现在从 source 车站出发（初始时不在公交车上），要前往 target 车站。 期间仅可乘坐公交车。

        求出 最少乘坐的公交车数量 。如果不可能到达终点车站，返回 -1 。

        输入：routes = [[1,2,7],[3,6,7]], source = 1, target = 6
        输出：2
        解释：最优策略是先乘坐第一辆公交车到达车站 7 , 然后换乘第二辆公交车到车站 6 。
    """

    def numBusesToDestination(self, routes: List[List[int]], source: int, target: int) -> int:
        if source == target:
            return 0

        end_buses = []
        sources = set()
        sources.add(source)
        temp = []
        t = 1

        for i in range(len(routes)):
            if target in routes[i]:
                end_buses.append(i)

        if len(end_buses) < 1:
            return -1

        while True:
            start_buses = []
            for i in range(len(routes)):
                if len(sources & set(routes[i])) > 0:
                    start_buses.append(i)

            if len(start_buses) < 1 or start_buses == temp:
                return -1

            for bus in start_buses:
                if bus in end_buses:
                    return t
                else:
                    sources.update(set(routes[bus]))
            t += 1
            temp = start_buses


# 1
class Solution1:
    """
        给定一个整数数组 nums和一个整数目标值 target，请你在该数组中找出 和为目标值 target 的那两个整数，并返回它们的数组下标。

        你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

        你可以按任意顺序返回答案。

        输入：nums = [2,7,11,15], target = 9
        输出：[0,1]
        解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
    """

    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nums_mapper = {v: k for k, v in enumerate(nums)}
        for i in range(len(nums) - 1):
            if target - nums[i] in nums_mapper.keys():
                if nums_mapper[target - nums[i]] != i:
                    return [i, nums_mapper[target - nums[i]]]
        return []


# 2
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution2:
    """
        给你两个非空 的链表，表示两个非负的整数。它们每位数字都是按照逆序的方式存储的，并且每个节点只能存储一位数字。

        请你将两个数相加，并以相同形式返回一个表示和的链表。

        你可以假设除了数字 0 之外，这两个数都不会以 0开头。

        输入：l1 = [2,4,3], l2 = [5,6,4]
        输出：[7,0,8]
        解释：342 + 465 = 807.
    """

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        times = 1
        current = l1
        result = 0
        while current:
            result = result + current.val * times
            current = current.next
            times *= 10

        times = 1
        current = l2
        while current:
            result = result + current.val * times
            current = current.next
            times *= 10

        result = str(result)
        length = len(result)
        next_val = None
        answer = None
        for i in range(length):
            answer = ListNode(int(result[i]), next_val)
            next_val = answer
        return answer


# 168
class Solution168:
    """
        给定一个正整数，返回它在 Excel 表中相对应的列名称。
        例如，
            1 -> A
            2 -> B
            3 -> C
            ...
            26 -> Z
            27 -> AA
            28 -> AB
            ...
        输入: 1
        输出: "A"
    """
    def convertToTitle(self, columnNumber: int) -> str:
        words = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        output = []
        val = columnNumber - 1
        while val >= 0:
            mod = val % 26
            output.append(words[mod])
            val = int(val / 26) - 1
        return ''.join(reversed(output))


# 3
class Solution3:
    """
        给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。

        输入: s = "abcabcbb"
        输出: 3
        解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
    """
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0

        length = len(s)
        output = 1
        for i in range(length):
            for j in range(i + output, length + 1):
                if j - i != len(set(s[i:j])):
                    break
                output = j - i
            i += 1
        return output


# 4
class Solution4:
    """
        给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。

        输入：nums1 = [1,3], nums2 = [2]
        输出：2.00000
        解释：合并数组 = [1,2,3] ，中位数 2
    """
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        pass


# LCP 7
class SolutionLCP7:
    """
        小朋友 A 在和 ta 的小伙伴们玩传信息游戏，游戏规则如下：

        有 n 名玩家，所有玩家编号分别为 0 ～ n-1，其中小朋友 A 的编号为 0
        每个玩家都有固定的若干个可传信息的其他玩家（也可能没有）。传信息的关系是单向的（比如 A 可以向 B 传信息，但 B 不能向 A 传信息）。
        每轮信息必须需要传递给另一个人，且信息可重复经过同一个人
        给定总玩家数 n，以及按 [玩家编号,对应可传递玩家编号] 关系组成的二维数组 relation。返回信息从小 A (编号 0 ) 经过 k 轮传递到编号为 n-1 的小伙伴处的方案数；若不能到达，返回 0。

        输入：n = 5, relation = [[0,2],[2,1],[3,4],[2,3],[1,4],[2,0],[0,4]], k = 3

        输出：3

        解释：信息从小 A 编号 0 处开始，经 3 轮传递，到达编号 4。共有 3 种方案，分别是 0->2->0->4， 0->2->1->4， 0->2->3->4。

    """
    def numWays(self, n: int, relation: List[List[int]], k: int) -> int:
        node = {0: 1}
        times = 1
        output = 0
        while times <= k:
            temp = dict()
            for link in relation:
                if link[0] in node:
                    if times == k:
                        if link[1] == n-1:
                            output += node[link[0]]
                    elif link[1] in temp:
                        temp[link[1]] += node[link[0]]
                    else:
                        temp[link[1]] = node[link[0]]
            if times == k:
                break
            node.clear()
            node |= temp
            times += 1
        return output


# 1515
class Solution1515:
    """
        一家快递公司希望在新城市建立新的服务中心。公司统计了该城市所有客户在二维地图上的坐标，并希望能够以此为依据为新的服务中心选址：使服务中心 到所有客户的欧几里得距离的总和最小 。

        给你一个数组 positions ，其中 positions[i] = [xi, yi] 表示第 i 个客户在二维地图上的位置，返回到所有客户的 欧几里得距离的最小总和 。

        换句话说，请你为服务中心选址，该位置的坐标 [xcentre, ycentre] 需要使下面的公式取到最小值：

        输入：positions = [[0,1],[1,0],[1,2],[2,1]]
        输出：4.00000
        解释：如图所示，你可以选 [xcentre, ycentre] = [1, 1] 作为新中心的位置，这样一来到每个客户的距离就都是 1，所有距离之和为 4 ，这也是可以找到的最小值。
    """
    def getMinDistSum(self, positions: List[List[int]]) -> float:
        s1 = 0
        s2 = 0
        t = 0
        for loc in positions:
            t += 1
            s1 += loc[0]
            s2 += loc[1]

        x = s1 / t
        y = s2 / t

        step = 1
        decay = 0.003
        accurary = 0.000000001
        output = 0
        while True:
            d1 = 0
            d2 = 0
            distance = 0

            for loc in positions:
                if (x - loc[0]) ** 2 + (y - loc[1]) ** 2 == 0:
                    d1 += 0
                    d2 += 0
                else:
                    d1 += (x - loc[0]) / (((x - loc[0]) ** 2 + (y - loc[1]) ** 2) ** 0.5)
                    d2 += (y - loc[1]) / (((x - loc[0]) ** 2 + (y - loc[1]) ** 2) ** 0.5)

            for loc in positions:
                distance += ((x - loc[0]) ** 2 + (y - loc[1]) ** 2) ** 0.5

            if abs(output - distance) < accurary:
                break
            output = distance

            x -= d1 * step
            y -= d2 * step
            step *= (1.0 - decay)

        return output


# 1914
class Solution1914:
    """
        给你一个大小为 m x n 的整数矩阵 grid，其中 m 和 n 都是 偶数 ；另给你一个整数 k 。

        矩阵由若干层组成，如下图所示，每种颜色代表一层：

        1 1 1 1
        1 2 2 1
        1 2 2 1
        1 1 1 1

        矩阵的循环轮转是通过分别循环轮转矩阵中的每一层完成的。在对某一层进行一次循环旋转操作时，层中的每一个元素将会取代其 逆时针 方向的相邻元素。

        返回执行 k 次循环轮转操作后的矩阵。

        40 10    10 20
        30 20    40 30

        输入：grid = [[40,10],[30,20]], k = 1
        输出：[[10,20],[40,30]]
        解释：上图展示了矩阵在执行循环轮转操作时每一步的状态。

    """
    def rotateGrid(self, grid: List[List[int]], k: int) -> List[List[int]]:
        m = len(grid)
        if m > 0:
            n = len(grid[0])
        else:
            return []
        level = int(min(m, n) / 2)
        output = [[0] * n for x in range(m)]

        for i in range(level):
            temp = list()

            temp.extend(grid[0 + i][0 + i:n - i])

            for j in range(i + 1, m - i - 1):
                temp.append(grid[j][n - i - 1])

            temp.extend(reversed(grid[m - i - 1][0 + i:n - i]))

            for j in range(i + 1, m - i - 1):
                temp.append(grid[m - j - 1][0 + i])

            times = len(temp)
            new_temp = temp[k % times:] + temp[0:k % times]

            for j in range(0 + i, n - i):
                output[0 + i][j] = new_temp.pop(0)

            for j in range(i + 1, m - i - 1):
                output[j][n - i - 1] = new_temp.pop(0)

            for j in range(0 + i, n - i):
                output[m - i - 1][n - 1 - j] = new_temp.pop(0)
            for j in range(i + 1, m - i - 1):
                output[m - j - 1][0 + i] = new_temp.pop(0)
        return output


# 930
class Solution930:
    """
        给你一个二元数组 nums ，和一个整数 goal ，请你统计并返回有多少个和为 goal 的 非空 子数组。

        子数组 是数组的一段连续部分。

        输入：nums = [1,0,1,0,1], goal = 2
        输出：4
        解释：
        有 4 个满足题目要求的子数组：[1,0,1]、[1,0,1,0]、[0,1,0,1]、[1,0,1]
    """
    def numSubarraysWithSum(self, nums: List[int], goal: int) -> int:
        length = len(nums)
        count = 0
        left, right = 0, 0
        sum1, sum2 = 0, 0
        if length < 1:
            return 0
        for i in range(length):
            sum1 += nums[i]
            while left <= i and sum1 > goal:
                sum1 -= nums[left]
                left += 1

            sum2 += nums[i]
            while right <= i and sum2 >= goal:
                sum2 -= nums[right]
                right += 1
            count += (right - left)
        return count


# 4
class Solution4:
    """
        给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
        输入：nums1 = [1,3], nums2 = [2]
        输出：2.00000
        解释：合并数组 = [1,2,3] ，中位数 2
    """
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        len1, len2 = len(nums1), len(nums2)
        num_arr = list()
        i, j = 0, 0
        while i < len1 and j < len2:
            if nums1[i] <= nums2[j]:
                num_arr.append(nums1[i])
                i += 1
            else:
                num_arr.append(nums2[j])
                j += 1
        num_arr.extend(nums1[i:])
        num_arr.extend(nums2[j:])
        if (len1 + len2) % 2 == 0:
            k = int((len1 + len2) / 2)
            return (num_arr[k] + num_arr[k - 1]) / 2.0
        else:
            k = int((len1 + len2 - 1) / 2)
            return num_arr[k]


# 面试题  17.10
class SolutionM17A10:
    """
        数组中占比超过一半的元素称之为主要元素。给你一个 整数 数组，找出其中的主要元素。若没有，返回 -1 。请设计时间复杂度为 O(N) 、空间复杂度为 O(1) 的解决方案。
        输入：[1,2,5,9,5,9,5,5,5]
        输出：5
    """
    def majorityElement(self, nums: List[int]) -> int:
        index, cursor = 0, 0
        for num in nums:
            if index == 0:
                cursor = num
                index = 1
            elif num == cursor:
                index += 1
            else:
                index -= 1
        count = 0
        for num in nums:
            if num == cursor:
                count += 1
        return cursor if count > int(len(nums) / 2) else -1


# 275
class Solution275:
    """
        给定一位研究者论文被引用次数的数组（被引用次数是非负整数），数组已经按照升序排列。编写一个方法，计算出研究者的 h 指数。

        h 指数的定义: “h 代表“高引用次数”（high citations），一名科研人员的 h 指数是指他（她）的 （N 篇论文中）总共有 h 篇论文分别被引用了至少 h 次。（其余的N - h篇论文每篇被引用次数不多于 h 次。）"
        输入: citations = [0,1,3,5,6]
        输出: 3
        解释: 给定数组表示研究者总共有 5 篇论文，每篇论文相应的被引用了 0, 1, 3, 5, 6 次。
             由于研究者有 3 篇论文每篇至少被引用了 3 次，其余两篇论文每篇被引用不多于 3 次，所以她的 h 指数是 3。

    """
    def hIndex(self, citations: List[int]) -> int:
        i = 0
        while i < len(citations):
            if citations[- i - 1] < i:
                break
        return i


# 218
class Solution218:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        if len(buildings) < 1:
            return []
        values = dict()
        values_end = dict()
        values_start = dict()
        for building in buildings:
            left = building[0]
            right = building[1]
            height = building[2]

            def _valid(key, value):
                if key in values and values[key] == value:
                    del values[key]
                elif key in values and values[key] > value:
                    pass
                else:
                    values[key] = value

            _valid(left, height)
            _valid(right, height)
            if right not in values_end:
                values_end[right] = set()
            values_end[right].add(height)
            if left not in values_start:
                values_start[left] = set()
            values_start[left].add(height)

        line_group: list = sorted(values.items(), key=lambda x: x[0])
        container = dict()

        def _max(c):
            if not c:
                return 0
            return max(c)
        point_group = list()
        for line in line_group:
            if line[0] in values_start:
                for start in values_start[line[0]]:
                    if start not in container:
                        container[start] = 0
                    container[start] += 1
                if line[1] >= _max(container):
                    point_group.append([line[0], line[1]])
            if line[0] in values_end:
                for end in values_end[line[0]]:
                    if end in container:
                        container[end] -= 1
                        if container[end] < 1:
                            del container[end]
                if line[1] > _max(container):
                    point_group.append([line[0], _max(container)])
        return point_group


# 1818
class Solution1818:
    """
        给你两个正整数数组 nums1 和 nums2 ，数组的长度都是 n 。

        数组 nums1 和 nums2 的 绝对差值和 定义为所有 |nums1[i] - nums2[i]|（0 <= i < n）的 总和（下标从 0 开始）。

        你可以选用 nums1 中的 任意一个 元素来替换 nums1 中的 至多 一个元素，以 最小化 绝对差值和。

        在替换数组 nums1 中最多一个元素 之后 ，返回最小绝对差值和。因为答案可能很大，所以需要对 10^9 + 7 取余 后返回。
    """
    def minAbsoluteSumDiff(self, nums1: List[int], nums2: List[int]) -> int:
        raw_abs = [abs(nums1[i] - nums2[i]) for i in range(len(nums1))]
        total_abs = sum(raw_abs)
        gap = 0
        while max_abs := max(raw_abs):
            cursor = raw_abs.index(max_abs)
            number = nums2[cursor]
            min_abs = max_abs

            for num in nums1:
                if abs(num - number) < min_abs:
                    min_abs = abs(num - number)

            raw_abs[cursor] = 0

            if (max_abs - min_abs) >= max(raw_abs) > 0:
                return (total_abs - max_abs + min_abs) % (10**9 + 7)
            elif (max_abs - min_abs) > gap:
                gap = max_abs - min_abs

        return (total_abs - gap) % (10**9 + 7)


# 1846
class Solution1846:
    """
        给你一个正整数数组arr。请你对 arr执行一些操作（也可以不进行任何操作），使得数组满足以下条件：

        arr中 第一个元素必须为1。
        任意相邻两个元素的差的绝对值 小于等于1，也就是说，对于任意的 1 <= i < arr.length（数组下标从 0 开始），都满足abs(arr[i] - arr[i - 1]) <= 1。abs(x)为x的绝对值。
        你可以执行以下 2 种操作任意次：

        减小 arr中任意元素的值，使其变为一个 更小的正整数。
        重新排列arr中的元素，你可以以任意顺序重新排列。
        请你返回执行以上操作后，在满足前文所述的条件下，arr中可能的 最大值。
    """
    def maximumElementAfterDecrementingAndRearranging(self, arr: List[int]) -> int:
        s_arr = sorted(arr)
        length = len(arr)
        if length < 2:
            return 1
        s_arr[0] = 1
        for i in range(1, length):
            if s_arr[i] > i + 1:
                s_arr[i] = i + 1
            elif s_arr[i] - s_arr[i-1] > 1:
                return s_arr[i-1] + length - i
        return s_arr[length-1]


# 剑指offer 53
class SolutionJZOffer53:
    """
        统计一个数字在排序数组中出现的次数。
    """
    def search(self, nums: List[int], target: int) -> int:
        s_nums = sorted(nums)
        count = 0
        for num in s_nums:
            if num < target:
                continue
            elif num == target:
                count += 1
            else:
                break
        return count


# 34
class Solution34:
    """
        给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

        如果数组中不存在目标值 target，返回 [-1, -1]。

        进阶：

        你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？
    """
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        length = len(nums)
        if not nums:
            return [-1, -1]
        left, right = 0, length-1
        while right - left > 1:
            middle = int((right + left) / 2)
            if nums[middle] > target:
                right = middle
            elif nums[middle] < target:
                left = middle
            else:
                for i in range(left, middle):
                    if nums[i] == target:
                        left = i
                        break
                else:
                    left = middle
                for i in range(middle, right + 1):
                    if nums[i] != target:
                        right = i - 1
                        break
                return [left, right]
        else:
            if nums[left] == nums[right] == target:
                return [left, right]
            elif nums[left] == target:
                return [left, left]
            elif nums[right] == target:
                return [right, right]
            else:
                return [-1, -1]


# 1838
class Solution1838:
    """
        元素的 频数 是该元素在一个数组中出现的次数。

        给你一个整数数组 nums 和一个整数 k 。在一步操作中，你可以选择 nums 的一个下标，并将该下标对应元素的值增加 1 。

        执行最多 k 次操作后，返回数组中最高频元素的 最大可能频数 。

        输入：nums = [1,2,4], k = 5
        输出：3
        解释：对第一个元素执行 3 次递增操作，对第二个元素执 2 次递增操作，此时 nums = [4,4,4] 。
        4 是数组中最高频元素，频数是 3 。

        来源：力扣（LeetCode）
        链接：https://leetcode-cn.com/problems/frequency-of-the-most-frequent-element
    """
    def maxFrequency(self, nums: List[int], k: int) -> int:
        length = len(nums)
        if length < 1:
            return 1

        # 首先从小到大快排一下
        s_nums = sorted(nums)
        # 移除截距
        c_nums = [x-s_nums[0] for x in s_nums]
        # 滑动窗口
        out = 0
        left, right = 0, 0
        target = 0
        while right < length:
            if target > k:
                target -= (c_nums[right] - c_nums[left])
                left += 1
                if left > right:
                    right = left
                    target = 0
            elif target <= k:
                if right - left > out:
                    out = right - left
                right += 1
                if right < length:
                    target += ((right - left) * (c_nums[right] - c_nums[right-1]))
                else:
                    break
        return out + 1


# 37
class Solution37:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        pass


# 1877
class Solution1877:
    """
        一个数对(a,b)的 数对和等于a + b。最大数对和是一个数对数组中最大的数对和。

        比方说，如果我们有数对(1,5)，(2,3)和(4,4)，最大数对和为max(1+5, 2+3, 4+4) = max(6, 5, 8) = 8。
        给你一个长度为 偶数n的数组nums，请你将 nums中的元素分成 n / 2个数对，使得：
        
        nums中每个元素恰好在 一个数对中，且
        最大数对和的值 最小。
        请你在最优数对划分的方案下，返回最小的 最大数对和。

        输入：nums = [3,5,2,3]
        输出：7
        解释：数组中的元素可以分为数对 (3,3) 和 (5,2) 。
        最大数对和为 max(3+3, 5+2) = max(6, 7) = 7 。
    """
    def minPairSum(self, nums: List[int]) -> int:
        n = len(nums)
        sort_nums = sorted(nums)
        output = 0
        for i in range(n):
            if sort_nums[i] + sort_nums[n - i - 1] > output:
                output = sort_nums[i] + sort_nums[n - i - 1]
        return output


class Solution:
    def canBeTypedWords(self, text: str, brokenLetters: str) -> int:
        words = text.split(' ')
        count = 0
        template = frozenset(brokenLetters)
        for w in words:
            if w and len(set(w) & template) < 1:
                count += 1
        return count

    def addRungs(self, rungs: List[int], dist: int) -> int:
        n = len(rungs)
        if n < 1:
            return 0
        count = int((rungs[0] - 0.1) / dist)
        for i in range(1, n):
            d = rungs[i] - rungs[i-1]
            if d > dist:
                count += int(d - 0.1/dist)
        return count

    def maxPoints(self, points: List[List[int]]) -> int:
        m = len(points)
        n = len(points[0])
        last_line = points[0]
        for i in range(1, m):
            temp = [0]*n
            max_left = 0
            max_right = 0

            for j in range(n):
                max_left = max(max_left - 1, last_line[j])
                temp[j] = max_left

            for j in range(n):
                max_right = max(max_right - 1, last_line[n-j-1])
                temp[n - j - 1] = max(temp[n - j - 1], max_right)

            last_line = [temp[j] + points[i][j] for j in range(len(temp))]
        return max(last_line)

    def maxGeneticDifference(self, parents: List[int], queries: List[List[int]]) -> List[int]:
        pass


# 剑指offer 52
class SolutionJZOffer52:
    """
        输入两个链表，找出它们的第一个公共节点。
        输入：intersectVal= 2, listA = [0,9,1,2,4], listB = [3,2,4], skipA = 3, skipB = 1
        输出：Reference of the node with value = 2
        输入解释：相交节点的值为 2 （注意，如果两个列表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [0,9,1,2,4]，链表 B 为 [3,2,4]。在 A 中，相交节点前有 3 个节点；在 B 中，相交节点前有 1 个节点。
    """
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if headA is None or headB is None:
            return None
        h1, h2 = headA, headB
        while h1 is not h2:
            h1 = headB if h1 is None else h1.next
            h2 = headA if h2 is None else h2.next

        return h1


# 138
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution138:
    # 标准想法
    def copyRandomList1(self, head: 'Node') -> 'Node':
        if not head:
            return None
        h1 = Node(head.val)

        cursor = head
        h2 = h1
        n = 0
        # 先建立next结构
        while cursor.next:
            h2.next = Node(cursor.next.val)
            h2 = h2.next
            cursor = cursor.next
            n += 1

        cursor = head
        h2 = h1
        # 再关联random结构
        while cursor:
            random_cursor = cursor.random
            if random_cursor is not None:
                j = 0
                while random_cursor.next:
                    random_cursor = random_cursor.next
                    j += 1
                d = n - j
                h3 = h1
                for k in range(d):
                    h3 = h3.next
                h2.random = h3
            else:
                h2.random = None

            h2 = h2.next
            cursor = cursor.next
        return h1

    # 插入法
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head:
            return None

        cursor = head
        # 将克隆得节点插在原节点后
        while cursor:
            temp = cursor.next
            cursor.next = Node(cursor.val)
            cursor.next.next = temp
            cursor = temp

        cursor = head
        # 对克隆节点依照源节点做random关联，并跳过下一个源节点
        while cursor:
            clone_cursor = cursor.next
            clone_cursor.random = cursor.random.next if cursor.random else None
            cursor = cursor.next.next
            clone_cursor.next = cursor.next if cursor else None

        return head.next


# 1893
class Solution1893:
    """
        给你一个二维整数数组ranges和两个整数left和right。每个ranges[i] = [starti, endi]表示一个从starti到endi的闭区间。

        如果闭区间[left, right]内每个整数都被ranges中至少一个区间覆盖，那么请你返回true，否则返回false。
        
        已知区间 ranges[i] = [starti, endi] ，如果整数 x 满足 starti <= x <= endi，那么我们称整数x被覆盖了。

        输入：ranges = [[1,2],[3,4],[5,6]], left = 2, right = 5
        输出：true
        解释：2 到 5 的每个整数都被覆盖了：
        - 2 被第一个区间覆盖。
        - 3 和 4 被第二个区间覆盖。
        - 5 被第三个区间覆盖。
    """
    def isCovered(self, ranges: List[List[int]], left: int, right: int) -> bool:
        n = len(ranges)
        if n < 1:
            return False

        i = left
        j = 0
        ranges = sorted(ranges, key=lambda x: x[0])
        while i <= right:
            while j < n:
                arr = ranges[j]
                if arr[0] <= i <= arr[1]:
                    i = arr[1] + 1
                    break
                j += 1
            else:
                return False
        return True


# 1713
class Solution1713:
    """
        给你一个数组target，包含若干 互不相同的整数，以及另一个整数数组arr，arr可能 包含重复元素。

        每一次操作中，你可以在 arr的任意位置插入任一整数。比方说，如果arr = [1,4,1,2]，那么你可以在中间添加 3得到[1,4,3,1,2]。你可以在数组最开始或最后面添加整数。
    
        请你返回 最少操作次数，使得target成为arr的一个子序列。
    
        一个数组的 子序列指的是删除原数组的某些元素（可能一个元素都不删除），同时不改变其余元素的相对顺序得到的数组。比方说，[2,7,4]是[4,2,3,7,2,1,4]的子序列（加粗元素），但[2,4,2]不是子序列。
    """
    def minOperations(self, target: List[int], arr: List[int]) -> int:
        target_n, arr_n = len(target), len(arr)
        target_set = {v: k for k, v in enumerate(target)}

        if target_n < 1:
            return 0
        container = list()
        for i in range(arr_n):
            if arr[i] in target_set:
                # hash找值
                index = target_set[arr[i]]
                # 二叉树获取 index按排序对应的节点位置。
                i = bisect.bisect_left(container, index)
                if i == len(container):
                    container.append(index)
                else:
                    container[i] = index
        return len(target) - len(container)


# 671
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution671:
    def findSecondMinimumValue(self, root: TreeNode) -> int:
        if not root:
            return -1
        num = root.val
        r = root
        container = set()
        self.findNode(num, r, container)
        return min(container) if len(container) > 0 else -1

    def findNode(self, num, node, container):
        if not node.left:
            return

        if node.left.val > num:
            container.add(node.left.val)
        else:
            self.findNode(num, node.left, container)

        if node.right.val > num:
            container.add(node.right.val)
        else:
            self.findNode(num, node.right, container)


# 863
class Solution863:
    """
        给定一个二叉树（具有根结点root），一个目标结点target，和一个整数值 K 。

        返回到目标结点 target 距离为 K 的所有结点的值的列表。 答案可以以任何顺序返回。
    """
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        container = list()
        # 获取父节点map
        parents = {root: None}

        self.parentMap(root, parents)
        t = target
        times = 0
        while t in parents and times <= k:
            self.findNode(t, k-times, container)
            last = t
            t = parents[t]
            if not t:
                break
            if t.left == last:
                t.left = None
            else:
                t.right = None
            times += 1
        return container

    def parentMap(self, cursor, container: dict):
        if cursor.left:
            container[cursor.left] = cursor
            self.parentMap(cursor.left, container)
        if cursor.right:
            container[cursor.right] = cursor
            self.parentMap(cursor.right, container)

    def findNode(self, target: TreeNode, times: int, container: list):
        if times == 0:
            container.append(target.val)
            return
        if target.left:
            self.findNode(target.left, times-1, container)
        if target.right:
            self.findNode(target.right, times-1, container)


# 171
class Solution171:
    """
        给定一个Excel表格中的列名称，返回其相应的列序号。
    """
    def titleToNumber(self, columnTitle: str) -> int:
        index = 0
        output = 0
        for i in range(len(columnTitle)):
            ch = columnTitle[-i - 1]
            output += (26**index) * (ord(ch) - ord('A') + 1)
            index += 1
        return output


if __name__ == '__main__':
    s = Solution171()
    print(s.titleToNumber('ZZY'))
