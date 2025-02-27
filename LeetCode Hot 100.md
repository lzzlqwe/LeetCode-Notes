# 哈希表

## [1.字母异位词分组](https://leetcode.cn/problems/group-anagrams/description/)
思路：创建一个HashMap，对每个字符串进行排序，排序后相同结果的归在同一个key下。  
代码：
```
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> mp = new HashMap<>();
        for(String str : strs){
            char[] carr = str.toCharArray();
            Arrays.sort(carr);
            String key = String.valueOf(carr);
            if(mp.containsKey(key))
                mp.get(key).add(str);
            else{
                List<String> list = new ArrayList<>();
                list.add(str);
                mp.put(key, list); 
            }
        }
        return new ArrayList<>(mp.values());
    }
}
```

## [2.最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/description/)
思路：nums用一个hash表来存放，遍历hash表，如果存在比他小1的值，则考虑比他小的值即可，跳过；如果不存在比他小1的值，则说明当前值可能是最长连续序列的起点。  
代码：
```
class Solution {
    public int longestConsecutive(int[] nums) {
        //nums用一个hash表来存放
        Set<Integer> set = new HashSet<>();
        for(int x : nums) 
            set.add(x);
        int maxLength = 0;
        //遍历hash表
        for(int x : set){
            if(set.contains(x-1)) //如果存在比他小1的值，则考虑比他小的值即可。
                continue;
            int t = x; //不存在比他小1的值，说明当前值可能是最长连续序列的起点
            while(set.contains(t))
                t++;
            maxLength = Math.max(maxLength, t - x); //更新答案
        }
        return maxLength;
    }
}
```

# 双指针

## [1.移动零](https://leetcode.cn/problems/move-zeroes/description/)
思路：使用双指针，左指针始终指向第一个0， 右指针指向left后的第一个非零元素（其中left到right-1都为0），进行元素交换后，双指针继续向后移动。  
代码：
```
class Solution {
    public void moveZeroes(int[] nums) {
        int n = nums.length;
        int left = 0; //始终指向第一个0
        while(left < n && nums[left] != 0)
            left++;
        int right = left; //指向left后的非零元素
        while(right < n && nums[right] == 0)
            right++;
        while(right < n){
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            while(right < n && nums[right] == 0)
                right++;
            while(left < n && nums[left] != 0)
                left++;
        }
    }
}
```

## [2.盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/description/)
思路：定义两个指针left和right，left从最左边开始，right从最右边开始，每次移动最矮的那条边（因为移动最矮边才可能在中间找到最优解），并记录储存的最大水量。  
代码：
```
class Solution {
    public int maxArea(int[] height) {
        int ans = 0;
        int left = 0;
        int right = height.length - 1;
        while(left < right){
            int s = (right - left) * Math.min(height[left], height[right]);
            ans = Math.max(ans, s);
            if(height[left] < height[right])
                left++;
            else
                right--;
        }
        return ans;
    }
}
```

## [3.三数之和](https://leetcode.cn/problems/3sum/description/)
思路：先排序，然后定义三个指针i，left和right，外层循环指针i从0遍历到倒数第三个位置（留两个位置给相向指针left和right），指针left从i+1开始，指针right从最右边开始。（如果题目要求结果不能重复，则三个指针都只指向第一个重复元素，绕过后续重复元素）  
代码：
```
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums); //排序
        List<List<Integer>> ans = new ArrayList<>();
        int n = nums.length;
        int i = 0;
        while(i < n - 2){
            int left = i + 1;
            int right = n - 1;
            while(left < right){
                if(nums[i] + nums[left] + nums[right] == 0){
                    ans.add(new ArrayList(Arrays.asList(nums[i], nums[left], nums[right])));
                    left++;
                    while(left < right && nums[left-1] == nums[left])//指向第一个重复的元素
                        left++;
                    right--;
                    while(left < right && nums[right] == nums[right+1])//指向第一个重复的元素
                        right--;
                }
                else if(nums[i] + nums[left] + nums[right] < 0)
                    left++;
                else
                    right--;
            }
            i++;
            while(i < n - 2 && nums[i-1] == nums[i]) //指向第一个重复的元素,绕过后续重复元素
                i++;
        }
        return ans;
    }
}
```

## [4.接雨水](https://leetcode.cn/problems/trapping-rain-water/description/)
思路：定义两个数组，前缀数组记录从左边到下标i的最大高度，后缀数组记录从右边到下标i的最大高度，有了这两个数组就可以计算各个下标i的盛水单位量。  
代码：
```
class Solution {
    public int trap(int[] height) {
        int n = height.length;
        int[] pre = new int[n];
        int[] post = new int[n];
        //求前缀数组和后缀数组
        int max = 0;
        for(int i = 0; i < n; i++){
            max = Math.max(max, height[i]);
            pre[i] = max;
        }
        max = 0;
        for(int i = n-1; i >= 0; i--){
            max = Math.max(max, height[i]);
            post[i] = max;
        }
        //根据前缀数组和后缀数组求答案
        int ans = 0;
        for(int i = 0; i < n; i++){
            ans += Math.min(pre[i], post[i]) - height[i];
        }
        return ans;
    }
}
```

# 滑动窗口

### 模板代码
```
int start = 0;
int end = 0;
while(end < n){ //外层循环扩展右边界，内层循环扩展左边界
    xxx;  //当前考虑的元素
    while(start <= end && 条件){
        xxx
        start++; //扩展左边界
    }
    end++; //扩展右边界
}
```

### 滑动窗口使用前提：  
1. 连续子数组。
2. 有单调性。比如元素均为正数。

## [1.无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/)
思路：套用滑动窗口模板  
代码：
```
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        char[] strArr = s.toCharArray();
        int n = strArr.length;
        int left = 0;
        int right = 0;
        int maxLength = 0;
        while(right < n){
            char c = strArr[right];
            while(left <= right && set.contains(c)){
                set.remove(strArr[left]);
                left++;
            }
            set.add(c);
            maxLength = Math.max(maxLength, right - left + 1);
            right++;
        }
        return maxLength;
    }
}
```

## [2.找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/)
思路：首先统计字符串p的各个字母出现的次数，然后套用滑动窗口模板，统计滑动窗口中各字符的次数，如果滑动窗口中当前字符的次数大于字符串p的字符次数，则移动滑动窗口左边。当滑动窗口的长度等于字符串p的长度时，即找到一个答案。之后继续查找。  
代码：
```
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> ans = new ArrayList<>();
        //统计字符串p的各个字母出现的次数
        char[] p_mp = new char[26];
        char[] pArr = p.toCharArray();
        for(char c : pArr){
            p_mp[c-'a']++;
        }
        //找到异位词
        char[] s_mp = new char[26];
        int left = 0;
        int right = 0;
        char[] sArr = s.toCharArray();
        int n = sArr.length;
        while(right < n){
            char c = sArr[right]; //考虑当前字符
            // s_mp.merge(c, 1, Integer::sum); 
            s_mp[c-'a']++; //添加到hashmap
            //如果滑动窗口中当前字符的次数大于字符串p的字符次数，则移动滑动窗口左边
            while(left <= right && s_mp[c-'a'] > p_mp[c-'a']){ 
                s_mp[sArr[left]-'a']--;
                left++;
            }
            if(right - left + 1 == p.length()) //找到答案
                ans.add(left);
            right++;
        }
        return ans;
    }
}
```

# 子串

## [1.和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/description/)
思路：前缀和pre[i]表示[0...i]的和。和为k的子数组nums[j, i]满足pre[i] - pre[j-1] = k，即pre[i] - k = pre[j-1]。因此转换思路，在for循环中，查找以i结尾的和为k的子数组个数时，只需计算前面有多少个和为pre[i] - k的子数组。  
代码：
```
class Solution {
    public int subarraySum(int[] nums, int k) {
        //前缀和pre[i]表示[0...i]的和
        //和为k的子数组nums[j, i]满足pre[i] - pre[j-1] = k，即pre[i] - k = pre[j-1]
        //因此for循环，查找以i结尾的和为k的子数组个数，只需计算前面有多少个和为pre[i] - k的子数组
        int ans = 0; //记录答案
        int pre = 0; //记录前缀和
        Map<Integer, Integer> mp = new HashMap<>(); //记录前缀和的出现个数
        mp.put(0, 1); //初始化，pre[-1] = 0;
        for(int i = 0; i < nums.length; i++){
            pre += nums[i]; //更新前缀和
            if(mp.containsKey(pre - k))
                ans += mp.get(pre - k);
            mp.merge(pre, 1, Integer::sum);
        }
        return ans;
    }
}
```

## [2.滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/description/)
思路：从左到右，维护一个单调队列，队首元素就是当前滑动窗口的最大值，详见代码注释。  
代码：
```
class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        List<Integer> dequeue = new LinkedList<>(); //单调队列
        for(int i = 0; i < k-1; i++){ //初始维护单调队列
            while(!dequeue.isEmpty() && nums[dequeue.getLast()] <= nums[i])
                dequeue.removeLast();
            dequeue.addLast(i); //存放下标而不是值
        }
        //移动滑动窗口得到答案
        int n = nums.length;
        int[] ans = new int[n-k+1];
        for(int i = k-1; i < n; i++){
            //维护单调队列
            while(!dequeue.isEmpty() && nums[dequeue.getLast()] <= nums[i])
                dequeue.removeLast();
            dequeue.addLast(i); //存放下标而不是值
            //检查队首是否已经超出滑动窗口了
            while(dequeue.getFirst() < i - k + 1)
                dequeue.removeFirst();
            //队首元素即为当前滑动窗口最大值
            ans[i-k+1] = nums[dequeue.getFirst()];
        }
        return ans;
    }
}
```


## [3.最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/description/)
思路：属于滑动窗口专题，和滑动窗口专题的第二题很相似。虽然存在双重循环，但时间复杂度为并不是O(n^2)，而是O(C·n)，其中C是一个常数。因为这里内层循环遍历的是52个大小写字母，是固定的常数。
代码：
```
class Solution {
    public String minWindow(String s, String t) {
        char[] t_mp = new char[128]; //t的哈希表
        char[] tArr = t.toCharArray();
        for(char c : tArr){
            t_mp[c - 'A']++;
        }
        char[] s_mp = new char[128]; //s的哈希表
        char[] sArr = s.toCharArray();
        int left = 0;
        int right = 0;
        int n = sArr.length;
        int[] ans = new int[]{0, 100001}; //记录最小覆盖子串的起始下标和长度
        while(right < n){
            char c = sArr[right]; //考虑当前字符
            s_mp[c - 'A']++; //加入hash表
            while(left <= right && isCover(t_mp, s_mp)){
                if(right - left + 1 < ans[1]){ //更新并记录答案
                    ans[0] = left;
                    ans[1] = right - left + 1;
                }
                s_mp[sArr[left] - 'A']--;
                left++;
            }
            right++;
        }
        if(ans[1] > 100000) return "";
        else return s.substring(ans[0], ans[0]+ans[1]);
    }

    //遍历52个字母，检查当前滑动窗口是否覆盖了字符串t
    boolean isCover(char[] t_mp, char[] s_mp){
        for(int i = 0; i < 26; i++) //大写字母
            if(t_mp[i] > s_mp[i])
                return false;
        for(int i = 0; i < 26; i++) //小写字母
            if(t_mp[i+'a'-'A'] > s_mp[i+'a'-'A'])
                return false;
        return true;
    }
}
```

# 普通数组

## [1.最大子数组和](https://leetcode.cn/problems/maximum-subarray/description/)

思路1：套用滑动窗口模板。  
代码：
```
class Solution {
    public int maxSubArray(int[] nums) {
        int ans = Integer.MIN_VALUE;
        int left = 0;
        int right = 0;
        int sum = 0;
        while(right < nums.length){
            sum += nums[right];
            ans = Math.max(sum, ans); //更新答案
            while(left <= right && sum <= 0){
                sum -= nums[left];
                left++;
            }
            right++;
        }
        return ans;
    }
}
```

思路2：前缀和做法。由于子数组的元素和等于两个前缀和的差。我们可以一边遍历数组计算前缀和，一边维护前缀和的最小值，当前的前缀和减去前缀和的最小值，就得到了以当前元素结尾的子数组和的最大值，用它来更新答案的最大值。   
代码：  
```
class Solution {
    public int maxSubArray(int[] nums) {
        int pre = 0; //记录以下标i结果的前缀和
        int min_pre = 0; //记录i之前的最小前缀和
        int ans = Integer.MIN_VALUE; //记录答案
        for(int i = 0; i < nums.length; i++){
            pre += nums[i]; //计算前缀和
            ans = Math.max(pre - min_pre, ans); //更新答案
            min_pre = Math.min(min_pre, pre); //更新最小前缀和
        }
        return ans;
    }
}
```

## [2.合并区间](https://leetcode.cn/problems/merge-intervals/description/)

思路：对所有区间按照左端点从小到大进行排序。for循环，如果不相交无法合并则加入答案，如果相交可以合并则更新答案。注意List到Array数组的转换代码。  
代码：
```
class Solution {
    public int[][] merge(int[][] intervals) {
        List<int[]> ans = new ArrayList<>();
        //按照区间左端点进行排序
        Arrays.sort(intervals, (p, q)->{
            return p[0] - q[0];
        });
        for(int i = 0; i < intervals.length; i++){
            if(i != 0 && intervals[i][0] <= ans.get(ans.size()-1)[1]){// 可以合并
                //更新右端点
                ans.get(ans.size()-1)[1] = Math.max(intervals[i][1], ans.get(ans.size()-1)[1]);
            }
            else ans.add(intervals[i]); // 不相交，无法合并
        }
        return ans.toArray(new int[ans.size()][]);
    }
}
```

## [3.轮转数组](https://leetcode.cn/problems/rotate-array/description/)

思路：反转整个数组，然后反转前k个元素，最后反转剩余元素，即可得出答案。注意k取模，防止k大于n。
代码：
```
class Solution {
    public void rotate(int[] nums, int k) {
        //一种做法是需要一个额外数组，考虑原地轮转的方法
        int n = nums.length;
        k = k % n; //k取模，防止k大于n
        reverse(nums, 0, n-1); //反转整个数组
        reverse(nums, 0, k-1); //反转前k个元素
        reverse(nums, k, n-1); //反转剩余元素
    }

    public void reverse(int[] nums, int l, int r){ //反转特定区间内的元素
        while(l <= r){
            int t = nums[l];
            nums[l] = nums[r];
            nums[r] = t;
            l++;
            r--;
        }
    }
}
```

## [4.除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/description/)

思路：利用前缀积和后缀积即可得到答案。  
代码：
```
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] pre = new int[n]; //pre[i]表示i之前所有元素的前缀积
        int[] post = new int[n]; //post[i]表示i之后所有元素的后缀积
        //计算前缀积和后缀积
        pre[0] = 1;
        for(int i = 1; i < n; i++)
            pre[i] = pre[i-1] * nums[i-1];
        post[n-1] = 1;
        for(int i = n-2; i >= 0; i--)
            post[i] = post[i+1] * nums[i+1];
        //计算答案
        int[] ans = new int[n];
        for(int i = 0; i < n; i++)
            ans[i] = pre[i] * post[i];
        return ans;
    }
}
```

# 矩阵

## [1.矩阵置零](https://leetcode.cn/problems/set-matrix-zeroes/description/)

思路：我们可以用两个标记数组分别记录每一行和每一列是否有零出现。    
代码：
```
class Solution {
    public void setZeroes(int[][] matrix) {
        int[] row = new int[matrix.length]; //记录每行是否有0
        int[] col = new int[matrix[0].length]; //记录每列是否有0
        //记录各行各列的0情况
        for(int i = 0; i < row.length; i++){
            for(int j = 0; j < col.length; j++){
                if(matrix[i][j] == 0){
                    row[i] = 1;
                    col[j] = 1;
                }
            }
        }
        //原地修改
        for(int i = 0; i < row.length; i++){
            for(int j = 0; j < col.length; j++){
                if(row[i] == 1 || col[j] == 1)
                    matrix[i][j] = 0;
            }
        }
    }
}
```

## [2.螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/description/)

思路：定义“右下左上”四个方向，初始为右方向，一直走，直到越界或者格子已访问，则转换下一个方向，重复此过程。 当走过所有格子则结束。   
代码：
```
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        int[][] dir = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};//右下左上
        int curDir = 0; //当前方向
        int row = matrix.length;
        int col = matrix[0].length;
        List<Integer> ans = new ArrayList<>();//答案
        int curX = 0; //当前坐标
        int curY = 0; //当前坐标
        int nums = 0; //表示走过的格子数
        ans.add(matrix[curX][curY]);//记录答案
        matrix[curX][curY] = 101; //标记当前位置已经走过
        nums++;
        while(nums < row * col){ //判断是否走完所有格子
            int nextX = curX + dir[curDir][0];
            int nextY = curY + dir[curDir][1];
            //判断下一个位置是否越界或者已经访问过了
            if(nextX < 0 || nextX >= row || nextY < 0 || nextY >= col || matrix[nextX][nextY] > 100){
                curDir = (curDir + 1) % 4; //修改方向
                nextX = curX + dir[curDir][0]; //重新走下一步
                nextY = curY + dir[curDir][1];
            }
            curX = nextX;
            curY = nextY;
            ans.add(matrix[curX][curY]);//记录答案
            matrix[curX][curY] = 101; //标记当前位置已经走过
            nums++;
        }
        return ans;
    }
}
```

## [3.旋转图像](https://leetcode.cn/problems/rotate-image/description/)

思路：先以左对角线进行翻转，再水平翻转。   
代码：
```
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        //1.先以左对角线进行翻转
        for(int i = 0; i < n; i++){
            for(int j = i; j < n; j++){
                int t = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = t;
            }
        }
        //2.再水平翻转
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n/2; j++){
                int t = matrix[i][j];
                matrix[i][j] = matrix[i][n-1-j];
                matrix[i][n-1-j] = t;
            }
        }
    }
}
```

## [4.搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/description/)

思路：右上角元素是二叉搜索树（灵神）。还有一种思路对每行进行二分查找。   
代码：
```
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        int n = matrix[0].length;
        //右上角元素是二叉搜索树
        int x = 0;
        int y = n - 1;
        while(x < m && y >= 0){
            if(matrix[x][y] == target)
                return true;
            else if(matrix[x][y] > target) //左子树
                y--;
            else //右子树
                x++;
        }
        return false;
    }
}
```

# 链表

## [1.相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/description/)

思路：设a链表长为x+z，b链表长为y+z，其中z是公共链部分。a走完x+z再走y，b走完y+z再走x，如果两个链表相交，则会刚好相遇。      
代码：
```
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode p1 = headA;
        ListNode p2 = headB;
        while(p1 != p2){
            if(p1 != null) p1 = p1.next;
            else p1 = headB;
            if(p2 != null) p2 = p2.next;
            else p2 = headA;
        }
        return p1;
    }
}
```

## [2.反转链表](https://leetcode.cn/problems/reverse-linked-list/description/)
思路：反转链表经典模板。使用两个指针pre和cur，反转结束后，从原来的链表上看，pre指向最末尾节点，cur指向后续的下一个节点，为null。  
代码：
```
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode cur = head;
        ListNode pre = null;
        while(cur != null)
        {
            ListNode nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }
}
```

## [3.回文链表](https://leetcode.cn/problems/palindrome-linked-list/description/)
思路：快慢指针模板找到链表的中间节点 + 反转链表。  
代码：
```
class Solution {
    public boolean isPalindrome(ListNode head) {
        //利用快慢指针找到链表的中间节点
        ListNode fast = head;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }
        //找到中间节点，反转总链表的后半部分
        ListNode tail = reverse(slow);
        //检查是否为回文链表
        while(tail != null){
            if(head.val != tail.val)
                return false;
            head = head.next;
            tail = tail.next;
        }
        return true;
    }

    public ListNode reverse(ListNode node){
        ListNode pre = null;
        ListNode cur = node;
        while(cur != null){
            ListNode nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }
}
```

## [4.环形链表](https://leetcode.cn/problems/linked-list-cycle/description/)
思路：快慢指针判断是否存在环。若fast指针达到结尾（使用模板判断，因为fast一次走两步），则不存在环。若存在环，则fast肯定会追上slow指针。    
代码：
```
public class Solution {
    public boolean hasCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
            if(fast == slow) //若存在环，快指针一定会追上慢指针
                return true;
        }
        return false; //没有环
    }
}
```

## [5.环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)
思路：  
![image](https://github.com/user-attachments/assets/4af5834a-f78d-4eec-a66f-979847bb3022)  
![image](https://github.com/user-attachments/assets/732720b3-6468-4103-9552-1414b62fd844)  

代码：
```
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
            if(fast == slow){ //快指针追上慢指针，存在环
                while(head != slow){
                    head = head.next;
                    slow = slow.next;
                }
                return head; //找到开始入环的第一个节点
            }
        }
        return null; //不存在环
    }
}
```

## [6.合并两个有序链表]()
思路：空间复杂度为O(1)。设置一个哨兵节点，cur指针指向哨兵。同时有list1和list2双指针，cur每次指向双指针较小的那一个，以此类推。    
代码：
```
class Solution {
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode preHead = new ListNode(); //哨兵节点
        ListNode cur = preHead;
        while(list1 != null && list2 != null){
            if(list1.val < list2.val){
                cur.next = list1;
                cur = cur.next;
                list1 = list1.next;
            }
            else{
                cur.next = list2;
                cur = cur.next;
                list2 = list2.next;
            }
        }
        if(list1 == null){
            cur.next = list2;
        }else{
            cur.next = list1;
        }
        return preHead.next;
    }
}
```
