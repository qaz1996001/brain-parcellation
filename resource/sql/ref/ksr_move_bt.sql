CREATE DATABASE  IF NOT EXISTS `mes_omi` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `mes_omi`;
-- MySQL dump 10.13  Distrib 5.7.12, for Win32 (AMD64)
--
-- Host: 127.0.0.1    Database: mes_omi
-- ------------------------------------------------------
-- Server version	5.5.5-10.2.7-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `ksr_move_bt`
--

DROP TABLE IF EXISTS `ksr_move_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `ksr_move_bt` (
  `report_time` datetime NOT NULL COMMENT 'tomorrow am 08:00 is real time data,others is history  -- 2017/06/26 lkchena\n輪三班制 We work on 3 shifts. \n上早班 on the day shift \n上夜班 on the night shift \n上大夜班 on the overnight shift\n',
  `cate` varchar(6) DEFAULT NULL COMMENT 'category:\nRT : real-time\nYES : yesterday\nS1: day shift\nS2: night shift\nS3: over-night shift',
  `lot_id` varchar(16) NOT NULL COMMENT 'lot 編碼加上屬性別 –> need conf',
  `cart_no` varchar(16) DEFAULT NULL COMMENT 'extend length to 12 -- 2020/03/02 lkchena\nwait design naming rule\n',
  `lot_size` int(11) DEFAULT NULL COMMENT '幾乎等於 cassette 數量, 不是全部都是 40, 因為有可能換班or料用完, 做不滿 40, 此時就會輸入數量 >> 對應的就是 cassette 數量變少(不是滿批)',
  `piece_size` int(11) DEFAULT NULL COMMENT 'piece total count',
  `cast_pcs_spec` int(11) DEFAULT NULL COMMENT '一個 cassette 可以裝多少個 part, 應該定義在 part table, 而不是 carrier 上\n<-- get data from wip table(from part table)',
  `pri` int(11) DEFAULT NULL COMMENT 'priority: 1~999\nSmaller has priority.\n\n1~10: SHR\n11~100: Important customer',
  `part_id` varchar(24) NOT NULL,
  `part_raw` varchar(16) DEFAULT NULL COMMENT '產品的原料: 旭宏 現場 習慣看 原料的編號 -- 2017/07/20 lkchena',
  `ope_no` varchar(7) NOT NULL COMMENT '2020/03/17 lkchena: for option flow, extend length to 7',
  `stage_id` varchar(16) DEFAULT NULL,
  `stage_name` varchar(36) DEFAULT NULL,
  `stage_order` varchar(3) DEFAULT NULL COMMENT 'ope_no 的前三碼(main step, not sub steps)',
  `claim_time` timestamp NULL DEFAULT NULL,
  `area_id` varchar(12) DEFAULT NULL,
  `area_name` varchar(24) DEFAULT NULL,
  `cust_id` varchar(12) DEFAULT NULL,
  `cust_name` varchar(36) DEFAULT NULL,
  `order_id` varchar(16) DEFAULT NULL COMMENT '2017/11/21 lkchena: additon for tracking order',
  `b_vendor_id` varchar(6) DEFAULT NULL COMMENT '當 s = "B" -- backup 時, 寫入 vendor_id',
  `b_vendor_name` varchar(24) DEFAULT NULL,
  `track_in_time` timestamp NULL DEFAULT NULL,
  `track_out_time` timestamp NULL DEFAULT NULL,
  `proc_start_time` timestamp NULL DEFAULT NULL,
  `proc_end_time` timestamp NULL DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`report_time`,`lot_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ksr_move_bt`
--

LOCK TABLES `ksr_move_bt` WRITE;
/*!40000 ALTER TABLE `ksr_move_bt` DISABLE KEYS */;
/*!40000 ALTER TABLE `ksr_move_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:28
